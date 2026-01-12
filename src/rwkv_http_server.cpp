#include <httplib.h>

#include "commondef.h"
#include "json.hpp"
#include "runtime.h"
#include "rwkv_http_server.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::json;

namespace {

struct StreamState {
    std::mutex mutex;
    std::condition_variable cv;
    std::deque<std::string> chunks;
    bool done = false;
};

static void stream_send(const std::shared_ptr<StreamState> & state, std::string chunk) {
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->chunks.emplace_back(std::move(chunk));
    }
    state->cv.notify_one();
}

struct StreamContext {
    std::shared_ptr<StreamState> state;
    std::string id;
    int64_t created = 0;
    std::string model_name;
    bool sent_role = false;
};

static thread_local StreamContext * tls_stream_ctx = nullptr;

static std::string sse_event(const json & payload) {
    return "data: " + payload.dump() + "\n\n";
}

static std::string sse_done() {
    return "data: [DONE]\n\n";
}

static int64_t unix_timestamp() {
    using namespace std::chrono;
    return duration_cast<seconds>(system_clock::now().time_since_epoch()).count();
}

static std::string make_id(const std::string & prefix) {
    static std::atomic<uint64_t> counter{0};
    return prefix + std::to_string(unix_timestamp()) + "-" + std::to_string(counter.fetch_add(1));
}

static void set_error_response(httplib::Response & res, int status, const std::string & message, const std::string & type) {
    res.status = status;
    json body = {
        {"error", {
            {"message", message},
            {"type", type},
            {"code", status}
        }}
    };
    res.set_content(body.dump(), "application/json; charset=utf-8");
}

static void apply_sampling_params(rwkvmobile::Runtime & runtime, int model_id, const json & body) {
    float temperature = runtime.get_temperature(model_id);
    int top_k = runtime.get_top_k(model_id);
    float top_p = runtime.get_top_p(model_id);
    if (body.contains("temperature")) {
        temperature = body.value("temperature", temperature);
    }
    if (body.contains("top_k")) {
        top_k = body.value("top_k", top_k);
    }
    if (body.contains("top_p")) {
        top_p = body.value("top_p", top_p);
    }
    runtime.set_sampler_params(model_id, temperature, top_k, top_p);

    float presence_penalty = runtime.get_presence_penalty(model_id);
    float frequency_penalty = runtime.get_frequency_penalty(model_id);
    float penalty_decay = runtime.get_penalty_decay(model_id);
    if (body.contains("presence_penalty")) {
        presence_penalty = body.value("presence_penalty", presence_penalty);
    }
    if (body.contains("frequency_penalty")) {
        frequency_penalty = body.value("frequency_penalty", frequency_penalty);
    }
    if (body.contains("penalty_decay")) {
        penalty_decay = body.value("penalty_decay", penalty_decay);
    }
    runtime.set_penalty_params(model_id, presence_penalty, frequency_penalty, penalty_decay);
}

static json build_timings(rwkvmobile::Runtime & runtime, int model_id, int prompt_tokens, int predicted_tokens) {
    double prefill_speed = runtime.get_avg_prefill_speed(model_id);
    double decode_speed = runtime.get_avg_decode_speed(model_id);
    double prompt_per_token_ms = prefill_speed > 0.0 ? (1000.0 / prefill_speed) : 0.0;
    double predicted_per_token_ms = decode_speed > 0.0 ? (1000.0 / decode_speed) : 0.0;

    return json{
        {"prompt_per_token_ms", prompt_per_token_ms},
        {"prompt_per_second", prefill_speed},
        {"predicted_n", predicted_tokens},
        {"predicted_per_token_ms", predicted_per_token_ms},
        {"predicted_per_second", decode_speed}
    };
}

static std::string strip_prompt_prefix(const std::string & full, const std::string & prompt) {
    if (full.size() >= prompt.size() && full.compare(0, prompt.size(), prompt) == 0) {
        return full.substr(prompt.size());
    }
    return full;
}

static bool extract_message_content(const json & message, std::string & content, std::string & error) {
    if (!message.contains("content")) {
        error = "message.content is required";
        return false;
    }
    const auto & content_value = message.at("content");
    if (content_value.is_string()) {
        content = content_value.get<std::string>();
        return true;
    }
    if (content_value.is_array()) {
        std::string combined;
        for (const auto & item : content_value) {
            if (item.is_object() && item.value("type", "") == "text" && item.contains("text") && item["text"].is_string()) {
                combined += item["text"].get<std::string>();
            }
        }
        if (!combined.empty()) {
            content = combined;
            return true;
        }
        error = "message.content array has no text items";
        return false;
    }
    error = "message.content must be string or array";
    return false;
}

static void completion_callback(const char *, const int, const char * next) {
    if (tls_stream_ctx == nullptr || next == nullptr || next[0] == '\0') {
        return;
    }
    json chunk = {
        {"id", tls_stream_ctx->id},
        {"object", "text_completion"},
        {"created", tls_stream_ctx->created},
        {"model", tls_stream_ctx->model_name},
        {"choices", json::array({
            {
                {"index", 0},
                {"text", std::string(next)},
                {"finish_reason", nullptr}
            }
        })}
    };
    stream_send(tls_stream_ctx->state, sse_event(chunk));
}

static void chat_callback(const char *, const int, const char * next) {
    if (tls_stream_ctx == nullptr || next == nullptr || next[0] == '\0') {
        return;
    }
    json delta = json::object();
    if (!tls_stream_ctx->sent_role) {
        delta["role"] = "assistant";
        tls_stream_ctx->sent_role = true;
    }
    delta["content"] = std::string(next);
    json chunk = {
        {"id", tls_stream_ctx->id},
        {"object", "chat.completion.chunk"},
        {"created", tls_stream_ctx->created},
        {"model", tls_stream_ctx->model_name},
        {"choices", json::array({
            {
                {"index", 0},
                {"delta", delta},
                {"finish_reason", nullptr}
            }
        })}
    };
    stream_send(tls_stream_ctx->state, sse_event(chunk));
}

} // namespace

namespace rwkvmobile {

struct RwkvHttpServer::Impl {
    Runtime * runtime = nullptr;
    int model_id = -1;
    HttpServerConfig config;
    httplib::Server server;
    std::thread thread;
    std::mutex generation_mutex;
    std::atomic<bool> running{false};

    Impl(Runtime * runtime, int model_id, HttpServerConfig config)
        : runtime(runtime), model_id(model_id), config(std::move(config)) {}
};

RwkvHttpServer::RwkvHttpServer(Runtime * runtime, int model_id, HttpServerConfig config)
    : impl_(std::make_unique<Impl>(runtime, model_id, std::move(config))) {}

RwkvHttpServer::~RwkvHttpServer() {
    stop();
    wait();
}

int RwkvHttpServer::start() {
    if (impl_->runtime == nullptr || impl_->model_id < 0) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    if (impl_->running.load()) {
        return RWKV_ERROR_RUNTIME;
    }

    if (impl_->config.threads <= 0) {
        impl_->config.threads = std::max(1u, std::thread::hardware_concurrency());
    }

    if (impl_->config.has_temperature || impl_->config.has_top_k || impl_->config.has_top_p) {
        float temperature = impl_->config.has_temperature ? impl_->config.temperature : impl_->runtime->get_temperature(impl_->model_id);
        int top_k = impl_->config.has_top_k ? impl_->config.top_k : impl_->runtime->get_top_k(impl_->model_id);
        float top_p = impl_->config.has_top_p ? impl_->config.top_p : impl_->runtime->get_top_p(impl_->model_id);
        impl_->runtime->set_sampler_params(impl_->model_id, temperature, top_k, top_p);
    }
    if (impl_->config.has_presence_penalty || impl_->config.has_frequency_penalty || impl_->config.has_penalty_decay) {
        float presence_penalty = impl_->config.has_presence_penalty ? impl_->config.presence_penalty : impl_->runtime->get_presence_penalty(impl_->model_id);
        float frequency_penalty = impl_->config.has_frequency_penalty ? impl_->config.frequency_penalty : impl_->runtime->get_frequency_penalty(impl_->model_id);
        float penalty_decay = impl_->config.has_penalty_decay ? impl_->config.penalty_decay : impl_->runtime->get_penalty_decay(impl_->model_id);
        impl_->runtime->set_penalty_params(impl_->model_id, presence_penalty, frequency_penalty, penalty_decay);
    }

    impl_->server.new_task_queue = [this] {
        return new httplib::ThreadPool(impl_->config.threads);
    };

    impl_->server.Get("/health", [](const httplib::Request &, httplib::Response & res) {
        res.set_content("{\"status\":\"ok\"}", "application/json; charset=utf-8");
    });

    impl_->server.Get("/v1/batch/supported_batch_sizes", [this](const httplib::Request &, httplib::Response & res) {
        auto sizes = impl_->runtime->get_supported_batch_sizes(impl_->model_id);
        json arr = json::array();
        for (int s : sizes) {
            arr.push_back(s);
        }
        json response = {
            {"supported_batch_sizes", arr},
            {"model", impl_->config.model_name}
        };
        res.set_content(response.dump(), "application/json; charset=utf-8");
    });

    impl_->server.Get("/v1/chat/roles", [this](const httplib::Request &, httplib::Response & res) {
        json response = {
            {"user_role", impl_->runtime->get_user_role(impl_->model_id)},
            {"assistant_role", impl_->runtime->get_response_role(impl_->model_id)},
            {"model", impl_->config.model_name}
        };
        res.set_content(response.dump(), "application/json; charset=utf-8");
    });

    impl_->server.Post("/v1/chat/roles", [this](const httplib::Request & req, httplib::Response & res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (const std::exception & e) {
            set_error_response(res, 400, std::string("invalid JSON: ") + e.what(), "invalid_request_error");
            return;
        }
        if (body.contains("user_role") && body["user_role"].is_string()) {
            impl_->runtime->set_user_role(impl_->model_id, body["user_role"].get<std::string>());
        }
        if (body.contains("assistant_role") && body["assistant_role"].is_string()) {
            impl_->runtime->set_response_role(impl_->model_id, body["assistant_role"].get<std::string>());
        }
        json response = {
            {"user_role", impl_->runtime->get_user_role(impl_->model_id)},
            {"assistant_role", impl_->runtime->get_response_role(impl_->model_id)},
            {"model", impl_->config.model_name}
        };
        res.set_content(response.dump(), "application/json; charset=utf-8");
    });

    impl_->server.Post("/v1/completions", [this](const httplib::Request & req, httplib::Response & res) {
        auto lock = std::make_shared<std::unique_lock<std::mutex>>(impl_->generation_mutex);
        json body;
        try {
            body = json::parse(req.body);
        } catch (const std::exception & e) {
            set_error_response(res, 400, std::string("invalid JSON: ") + e.what(), "invalid_request_error");
            return;
        }

        if (!body.contains("prompt") || !body["prompt"].is_string()) {
            set_error_response(res, 400, "prompt is required", "invalid_request_error");
            return;
        }
        std::string prompt = body["prompt"].get<std::string>();
        int prompt_tokens = (int)impl_->runtime->tokenizer_encode(impl_->model_id, prompt).size();
        int max_tokens = body.value("max_tokens", impl_->config.default_max_tokens);
        if (max_tokens <= 0) {
            set_error_response(res, 400, "max_tokens must be positive", "invalid_request_error");
            return;
        }
        int stop_code = body.value("stop_code", 0);

        apply_sampling_params(*impl_->runtime, impl_->model_id, body);

        bool stream = body.value("stream", false);
        if (!stream) {
            int ret = impl_->runtime->gen_completion(impl_->model_id, prompt, max_tokens, stop_code, nullptr);
            if (ret != rwkvmobile::RWKV_SUCCESS) {
                set_error_response(res, 500, "generation failed", "server_error");
                return;
            }

            std::string full_text = impl_->runtime->get_response_buffer_content(impl_->model_id);
            std::string completion_text = strip_prompt_prefix(full_text, prompt);
            int predicted_tokens = (int)impl_->runtime->get_response_buffer_ids(impl_->model_id).size() - prompt_tokens;
            predicted_tokens = std::max(0, predicted_tokens);
            json timings = build_timings(*impl_->runtime, impl_->model_id, prompt_tokens, predicted_tokens);
            json response = {
                {"id", make_id("cmpl-")},
                {"object", "text_completion"},
                {"created", unix_timestamp()},
                {"model", impl_->config.model_name},
                {"choices", json::array({
                    {
                        {"index", 0},
                        {"text", completion_text},
                        {"finish_reason", "stop"}
                    }
                })},
                {"timings", timings}
            };
            res.set_content(response.dump(), "application/json; charset=utf-8");
            return;
        }

        const std::string id = make_id("cmpl-");
        const int64_t created = unix_timestamp();
        auto state = std::make_shared<StreamState>();
        auto ctx = std::make_shared<StreamContext>();
        ctx->state = state;
        ctx->id = id;
        ctx->created = created;
        ctx->model_name = impl_->config.model_name;

        std::thread worker([this, ctx, state, id, created, prompt, prompt_tokens, max_tokens, stop_code]() {
            tls_stream_ctx = ctx.get();
            int ret = impl_->runtime->gen_completion(impl_->model_id, prompt, max_tokens, stop_code, completion_callback);
            tls_stream_ctx = nullptr;
            int predicted_tokens = (int)impl_->runtime->get_response_buffer_ids(impl_->model_id).size() - prompt_tokens;
            predicted_tokens = std::max(0, predicted_tokens);
            json timings = build_timings(*impl_->runtime, impl_->model_id, prompt_tokens, predicted_tokens);
            json final_chunk = {
                {"id", id},
                {"object", "text_completion"},
                {"created", created},
                {"model", impl_->config.model_name},
                {"choices", json::array({
                    {
                        {"index", 0},
                        {"text", ""},
                        {"finish_reason", ret == rwkvmobile::RWKV_SUCCESS ? "stop" : "error"}
                    }
                })},
                {"timings", timings}
            };
            stream_send(state, sse_event(final_chunk));
            stream_send(state, sse_done());
            {
                std::lock_guard<std::mutex> lock_state(state->mutex);
                state->done = true;
            }
            state->cv.notify_one();
        });
        worker.detach();

        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");
        res.set_chunked_content_provider(
            "text/event-stream",
            [state, lock](size_t, httplib::DataSink & sink) -> bool {
                std::unique_lock<std::mutex> lock_state(state->mutex);
                state->cv.wait(lock_state, [&state]() {
                    return !state->chunks.empty() || state->done;
                });
                if (!state->chunks.empty()) {
                    std::string chunk = std::move(state->chunks.front());
                    state->chunks.pop_front();
                    lock_state.unlock();
                    sink.write(chunk.data(), chunk.size());
                    return true;
                }
                return false;
            },
            [state, lock](bool) mutable {
                state.reset();
                lock.reset();
            }
        );
    });

    impl_->server.Post("/v1/chat/completions", [this](const httplib::Request & req, httplib::Response & res) {
        auto lock = std::make_shared<std::unique_lock<std::mutex>>(impl_->generation_mutex);
        json body;
        try {
            body = json::parse(req.body);
        } catch (const std::exception & e) {
            set_error_response(res, 400, std::string("invalid JSON: ") + e.what(), "invalid_request_error");
            return;
        }

        if (!body.contains("messages") || !body["messages"].is_array()) {
            set_error_response(res, 400, "messages is required", "invalid_request_error");
            return;
        }

        std::vector<std::string> inputs;
        std::vector<std::string> roles;
        for (const auto & message : body["messages"]) {
            if (!message.is_object()) {
                set_error_response(res, 400, "each message must be an object", "invalid_request_error");
                return;
            }
            std::string content;
            std::string error;
            if (!extract_message_content(message, content, error)) {
                set_error_response(res, 400, error, "invalid_request_error");
                return;
            }
            std::string role = message.value("role", "user");
            inputs.push_back(content);
            roles.push_back(role);
        }
        if (inputs.empty()) {
            set_error_response(res, 400, "messages cannot be empty", "invalid_request_error");
            return;
        }

        int max_tokens = body.value("max_tokens", impl_->config.default_max_tokens);
        if (max_tokens <= 0) {
            set_error_response(res, 400, "max_tokens must be positive", "invalid_request_error");
            return;
        }

        bool enable_reasoning = body.value("enable_reasoning", false);
        bool force_reasoning = body.value("force_reasoning", false);
        int force_lang = body.value("force_lang", 0);
        if (force_lang == 0) {
            std::string force_language = body.value("force_language", "");
            if (force_language == "zh" || force_language == "zh-CN") {
                force_lang = 1;
            }
        }

        std::string prompt_text = impl_->runtime->apply_chat_template(impl_->model_id, inputs, enable_reasoning, true, roles);
        int prompt_tokens = (int)impl_->runtime->tokenizer_encode(impl_->model_id, prompt_text).size();

        apply_sampling_params(*impl_->runtime, impl_->model_id, body);

        bool stream = body.value("stream", false);
        if (!stream) {
            int ret = impl_->runtime->chat(impl_->model_id, inputs, max_tokens, nullptr, enable_reasoning, force_reasoning, true, force_lang, roles);
            if (ret != rwkvmobile::RWKV_SUCCESS) {
                set_error_response(res, 500, "generation failed", "server_error");
                return;
            }

            std::string response_text = impl_->runtime->get_response_buffer_content(impl_->model_id);
            int predicted_tokens = (int)impl_->runtime->get_response_buffer_ids(impl_->model_id).size();
            json timings = build_timings(*impl_->runtime, impl_->model_id, prompt_tokens, predicted_tokens);
            json response = {
                {"id", make_id("chatcmpl-")},
                {"object", "chat.completion"},
                {"created", unix_timestamp()},
                {"model", impl_->config.model_name},
                {"choices", json::array({
                    {
                        {"index", 0},
                        {"message", {
                            {"role", "assistant"},
                            {"content", response_text}
                        }},
                        {"finish_reason", "stop"}
                    }
                })},
                {"timings", timings}
            };
            res.set_content(response.dump(), "application/json; charset=utf-8");
            return;
        }

        const std::string id = make_id("chatcmpl-");
        const int64_t created = unix_timestamp();
        auto state = std::make_shared<StreamState>();
        auto ctx = std::make_shared<StreamContext>();
        ctx->state = state;
        ctx->id = id;
        ctx->created = created;
        ctx->model_name = impl_->config.model_name;
        ctx->sent_role = false;

        std::thread worker([this, ctx, state, id, created, inputs, roles, prompt_tokens, max_tokens, enable_reasoning, force_reasoning, force_lang]() {
            tls_stream_ctx = ctx.get();
            int ret = impl_->runtime->chat(impl_->model_id, inputs, max_tokens, chat_callback, enable_reasoning, force_reasoning, true, force_lang, roles);
            tls_stream_ctx = nullptr;
            int predicted_tokens = (int)impl_->runtime->get_response_buffer_ids(impl_->model_id).size();
            json timings = build_timings(*impl_->runtime, impl_->model_id, prompt_tokens, predicted_tokens);
            json final_chunk = {
                {"id", id},
                {"object", "chat.completion.chunk"},
                {"created", created},
                {"model", impl_->config.model_name},
                {"choices", json::array({
                    {
                        {"index", 0},
                        {"delta", json::object()},
                        {"finish_reason", ret == rwkvmobile::RWKV_SUCCESS ? "stop" : "error"}
                    }
                })},
                {"timings", timings}
            };
            stream_send(state, sse_event(final_chunk));
            stream_send(state, sse_done());
            {
                std::lock_guard<std::mutex> lock_state(state->mutex);
                state->done = true;
            }
            state->cv.notify_one();
        });
        worker.detach();

        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");
        res.set_chunked_content_provider(
            "text/event-stream",
            [state, lock](size_t, httplib::DataSink & sink) -> bool {
                std::unique_lock<std::mutex> lock_state(state->mutex);
                state->cv.wait(lock_state, [&state]() {
                    return !state->chunks.empty() || state->done;
                });
                if (!state->chunks.empty()) {
                    std::string chunk = std::move(state->chunks.front());
                    state->chunks.pop_front();
                    lock_state.unlock();
                    sink.write(chunk.data(), chunk.size());
                    return true;
                }
                return false;
            },
            [state, lock](bool) mutable {
                state.reset();
                lock.reset();
            }
        );
    });

    impl_->server.Post("/v1/batch/completions", [this](const httplib::Request & req, httplib::Response & res) {
        auto lock = std::make_shared<std::unique_lock<std::mutex>>(impl_->generation_mutex);
        json body;
        try {
            body = json::parse(req.body);
        } catch (const std::exception & e) {
            set_error_response(res, 400, std::string("invalid JSON: ") + e.what(), "invalid_request_error");
            return;
        }

        if (!body.contains("prompts") || !body["prompts"].is_array()) {
            set_error_response(res, 400, "prompts is required", "invalid_request_error");
            return;
        }
        std::vector<std::string> prompts;
        std::vector<int> prompt_tokens_batch;
        for (const auto & item : body["prompts"]) {
            if (!item.is_string()) {
                set_error_response(res, 400, "prompts must be array of strings", "invalid_request_error");
                return;
            }
            std::string prompt = item.get<std::string>();
            prompts.push_back(prompt);
            prompt_tokens_batch.push_back((int)impl_->runtime->tokenizer_encode(impl_->model_id, prompt).size());
        }
        if (prompts.empty()) {
            set_error_response(res, 400, "prompts cannot be empty", "invalid_request_error");
            return;
        }

        int max_tokens = body.value("max_tokens", impl_->config.default_max_tokens);
        if (max_tokens <= 0) {
            set_error_response(res, 400, "max_tokens must be positive", "invalid_request_error");
            return;
        }
        int stop_code = body.value("stop_code", 0);

        apply_sampling_params(*impl_->runtime, impl_->model_id, body);

        int batch_size = (int)prompts.size();
        int ret = impl_->runtime->gen_completion_batch(impl_->model_id, prompts, batch_size, max_tokens, stop_code, nullptr);
        if (ret != rwkvmobile::RWKV_SUCCESS) {
            set_error_response(res, 500, "generation failed", "server_error");
            return;
        }

        auto outputs = impl_->runtime->get_response_buffer_content_batch(impl_->model_id);
        auto output_ids = impl_->runtime->get_response_buffer_ids_batch(impl_->model_id);

        json choices = json::array();
        json timings = json::array();
        for (int i = 0; i < batch_size; ++i) {
            std::string full_text = i < (int)outputs.size() ? outputs[i] : "";
            std::string completion_text = strip_prompt_prefix(full_text, prompts[i]);
            int predicted_tokens = 0;
            if (i < (int)output_ids.size()) {
                predicted_tokens = (int)output_ids[i].size() - prompt_tokens_batch[i];
                predicted_tokens = std::max(0, predicted_tokens);
            }
            choices.push_back({
                {"index", i},
                {"text", completion_text},
                {"finish_reason", "stop"}
            });
            timings.push_back(build_timings(*impl_->runtime, impl_->model_id, prompt_tokens_batch[i], predicted_tokens));
        }

        json response = {
            {"id", make_id("cmpl-batch-")},
            {"object", "batch.completion"},
            {"created", unix_timestamp()},
            {"model", impl_->config.model_name},
            {"choices", choices},
            {"timings", timings}
        };
        res.set_content(response.dump(), "application/json; charset=utf-8");
    });

    impl_->server.Post("/v1/batch/chat", [this](const httplib::Request & req, httplib::Response & res) {
        auto lock = std::make_shared<std::unique_lock<std::mutex>>(impl_->generation_mutex);
        json body;
        try {
            body = json::parse(req.body);
        } catch (const std::exception & e) {
            set_error_response(res, 400, std::string("invalid JSON: ") + e.what(), "invalid_request_error");
            return;
        }

        if (!body.contains("conversations") || !body["conversations"].is_array()) {
            set_error_response(res, 400, "conversations is required", "invalid_request_error");
            return;
        }

        int max_tokens = body.value("max_tokens", impl_->config.default_max_tokens);
        if (max_tokens <= 0) {
            set_error_response(res, 400, "max_tokens must be positive", "invalid_request_error");
            return;
        }

        bool enable_reasoning = body.value("enable_reasoning", false);
        bool force_reasoning = body.value("force_reasoning", false);
        int force_lang = body.value("force_lang", 0);
        if (force_lang == 0) {
            std::string force_language = body.value("force_language", "");
            if (force_language == "zh" || force_language == "zh-CN") {
                force_lang = 1;
            }
        }

        std::vector<std::vector<std::string>> inputs_batch;
        std::vector<std::vector<std::string>> roles_batch;
        std::vector<int> prompt_tokens_batch;
        for (const auto & conv : body["conversations"]) {
            if (!conv.is_object() || !conv.contains("messages") || !conv["messages"].is_array()) {
                set_error_response(res, 400, "each conversation must contain messages array", "invalid_request_error");
                return;
            }
            std::vector<std::string> inputs;
            std::vector<std::string> roles;
            for (const auto & message : conv["messages"]) {
                if (!message.is_object()) {
                    set_error_response(res, 400, "each message must be an object", "invalid_request_error");
                    return;
                }
                std::string content;
                std::string error;
                if (!extract_message_content(message, content, error)) {
                    set_error_response(res, 400, error, "invalid_request_error");
                    return;
                }
                std::string role = message.value("role", "user");
                inputs.push_back(content);
                roles.push_back(role);
            }
            if (inputs.empty()) {
                set_error_response(res, 400, "messages cannot be empty", "invalid_request_error");
                return;
            }
            std::string prompt_text = impl_->runtime->apply_chat_template(impl_->model_id, inputs, enable_reasoning, true, roles);
            int prompt_tokens = (int)impl_->runtime->tokenizer_encode(impl_->model_id, prompt_text).size();
            inputs_batch.push_back(std::move(inputs));
            roles_batch.push_back(std::move(roles));
            prompt_tokens_batch.push_back(prompt_tokens);
        }

        if (inputs_batch.empty()) {
            set_error_response(res, 400, "conversations cannot be empty", "invalid_request_error");
            return;
        }

        apply_sampling_params(*impl_->runtime, impl_->model_id, body);

        int batch_size = (int)inputs_batch.size();
        int ret = impl_->runtime->chat_batch(impl_->model_id, inputs_batch, max_tokens, batch_size, nullptr, enable_reasoning, force_reasoning, true, force_lang, roles_batch);
        if (ret != rwkvmobile::RWKV_SUCCESS) {
            set_error_response(res, 500, "generation failed", "server_error");
            return;
        }

        auto outputs = impl_->runtime->get_response_buffer_content_batch(impl_->model_id);
        auto output_ids = impl_->runtime->get_response_buffer_ids_batch(impl_->model_id);

        json choices = json::array();
        json timings = json::array();
        for (int i = 0; i < batch_size; ++i) {
            std::string response_text = i < (int)outputs.size() ? outputs[i] : "";
            int predicted_tokens = 0;
            if (i < (int)output_ids.size()) {
                predicted_tokens = (int)output_ids[i].size();
            }
            choices.push_back({
                {"index", i},
                {"message", {
                    {"role", "assistant"},
                    {"content", response_text}
                }},
                {"finish_reason", "stop"}
            });
            timings.push_back(build_timings(*impl_->runtime, impl_->model_id, prompt_tokens_batch[i], predicted_tokens));
        }

        json response = {
            {"id", make_id("chat-batch-")},
            {"object", "batch.chat.completion"},
            {"created", unix_timestamp()},
            {"model", impl_->config.model_name},
            {"choices", choices},
            {"timings", timings}
        };
        res.set_content(response.dump(), "application/json; charset=utf-8");
    });

    bool was_bound = false;
    if (impl_->config.port == 0) {
        int bound_port = impl_->server.bind_to_any_port(impl_->config.host);
        was_bound = (bound_port >= 0);
        if (was_bound) {
            impl_->config.port = bound_port;
        }
    } else {
        was_bound = impl_->server.bind_to_port(impl_->config.host, impl_->config.port);
    }

    if (!was_bound) {
        return RWKV_ERROR_IO;
    }

    impl_->thread = std::thread([this]() {
        impl_->running.store(true);
        impl_->server.listen_after_bind();
        impl_->running.store(false);
    });

    return RWKV_SUCCESS;
}

int RwkvHttpServer::stop() {
    if (!impl_->running.load()) {
        return RWKV_SUCCESS;
    }
    impl_->server.stop();
    return RWKV_SUCCESS;
}

int RwkvHttpServer::wait() {
    if (impl_->thread.joinable()) {
        impl_->thread.join();
    }
    return RWKV_SUCCESS;
}

bool RwkvHttpServer::is_running() const {
    return impl_->running.load();
}

} // namespace rwkvmobile
