#include "runtime.h"
#include "commondef.h"
#include "c_api.h"
#include "logger.h"
#include "soc_detect.h"
#ifdef ENABLE_SERVER
#include "rwkv_http_server.h"
#endif
#include <memory>
#include <cstring>
#include <cstdlib>
#include <thread>

#ifdef ENABLE_WEBRWKV
#include "web_rwkv_ffi.h"
#endif

namespace rwkvmobile {

extern "C" {

struct rwkvmobile_server_config rwkvmobile_server_config_default() {
    struct rwkvmobile_server_config cfg;
    cfg.host = "0.0.0.0";
    cfg.port = 8000;
    cfg.threads = 0;
    cfg.model_name = "rwkv";
    cfg.default_max_tokens = 256;
    cfg.temperature = 1.0f;
    cfg.top_k = 1;
    cfg.top_p = 1.0f;
    cfg.presence_penalty = 0.0f;
    cfg.frequency_penalty = 0.0f;
    cfg.penalty_decay = 0.0f;
    cfg.has_temperature = 0;
    cfg.has_top_k = 0;
    cfg.has_top_p = 0;
    cfg.has_presence_penalty = 0;
    cfg.has_frequency_penalty = 0;
    cfg.has_penalty_decay = 0;
    return cfg;
}

#ifdef ENABLE_SERVER
struct RwkvServerHandle {
    std::unique_ptr<rwkvmobile::RwkvHttpServer> server;
};

static rwkvmobile::HttpServerConfig convert_server_config(const struct rwkvmobile_server_config * config) {
    struct rwkvmobile_server_config cfg = rwkvmobile_server_config_default();
    if (config != nullptr) {
        cfg = *config;
    }
    rwkvmobile::HttpServerConfig out;
    out.host = cfg.host != nullptr ? cfg.host : "0.0.0.0";
    out.port = cfg.port;
    out.threads = cfg.threads;
    out.model_name = cfg.model_name != nullptr ? cfg.model_name : "rwkv";
    out.default_max_tokens = cfg.default_max_tokens;
    out.temperature = cfg.temperature;
    out.top_k = cfg.top_k;
    out.top_p = cfg.top_p;
    out.presence_penalty = cfg.presence_penalty;
    out.frequency_penalty = cfg.frequency_penalty;
    out.penalty_decay = cfg.penalty_decay;
    out.has_temperature = cfg.has_temperature != 0;
    out.has_top_k = cfg.has_top_k != 0;
    out.has_top_p = cfg.has_top_p != 0;
    out.has_presence_penalty = cfg.has_presence_penalty != 0;
    out.has_frequency_penalty = cfg.has_frequency_penalty != 0;
    out.has_penalty_decay = cfg.has_penalty_decay != 0;
    return out;
}
#endif

rwkvmobile_runtime_t rwkvmobile_runtime_init() {
    Runtime * rt = new Runtime();
    return rt;
}

int rwkvmobile_runtime_release(rwkvmobile_runtime_t handle) {
    if (handle == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class Runtime *>(handle);
    int ret = rt->release();
    delete rt;
    return ret;
}

rwkvmobile_server_t rwkvmobile_server_start(rwkvmobile_runtime_t runtime, int model_id, const struct rwkvmobile_server_config * config) {
    if (runtime == nullptr || model_id < 0) {
        return nullptr;
    }
#ifdef ENABLE_SERVER
    auto rt = static_cast<class Runtime *>(runtime);
    auto cfg = convert_server_config(config);
    auto handle = std::make_unique<RwkvServerHandle>();
    handle->server = std::make_unique<rwkvmobile::RwkvHttpServer>(rt, model_id, cfg);
    int ret = handle->server->start();
    if (ret != RWKV_SUCCESS) {
        return nullptr;
    }
    return handle.release();
#else
    (void)runtime;
    (void)model_id;
    (void)config;
    return nullptr;
#endif
}

int rwkvmobile_server_stop(rwkvmobile_server_t server) {
    if (server == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
#ifdef ENABLE_SERVER
    auto handle = static_cast<RwkvServerHandle *>(server);
    return handle->server->stop();
#else
    return RWKV_ERROR_UNSUPPORTED;
#endif
}

int rwkvmobile_server_wait(rwkvmobile_server_t server) {
    if (server == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
#ifdef ENABLE_SERVER
    auto handle = static_cast<RwkvServerHandle *>(server);
    return handle->server->wait();
#else
    return RWKV_ERROR_UNSUPPORTED;
#endif
}

int rwkvmobile_server_release(rwkvmobile_server_t server) {
    if (server == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
#ifdef ENABLE_SERVER
    auto handle = static_cast<RwkvServerHandle *>(server);
    handle->server->stop();
    handle->server->wait();
    delete handle;
    return RWKV_SUCCESS;
#else
    return RWKV_ERROR_UNSUPPORTED;
#endif
}

int rwkvmobile_runtime_load_model(rwkvmobile_runtime_t handle, const char * model_path, const char * backend_name, const char * tokenizer_path) {
    return rwkvmobile_runtime_load_model_with_extra(handle, model_path, backend_name, tokenizer_path, nullptr);
}

int rwkvmobile_runtime_load_model_with_extra(rwkvmobile_runtime_t handle, const char * model_path, const char * backend_name, const char * tokenizer_path, void * extra) {
    if (handle == nullptr || model_path == nullptr || backend_name == nullptr || tokenizer_path == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class Runtime *>(handle);
    return rt->load_model(model_path, backend_name, tokenizer_path, extra);
}

int rwkvmobile_runtime_load_model_async(rwkvmobile_runtime_t handle, const char * model_path, const char * backend_name, const char * tokenizer_path) {
    return rwkvmobile_runtime_load_model_with_extra_async(handle, model_path, backend_name, tokenizer_path, nullptr);
}

int rwkvmobile_runtime_load_model_with_extra_async(rwkvmobile_runtime_t handle, const char * model_path, const char * backend_name, const char * tokenizer_path, void * extra) {
    if (handle == nullptr || model_path == nullptr || backend_name == nullptr || tokenizer_path == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class Runtime *>(handle);
    if (rt->is_loading_model()) {
        LOGE("Model is already loading");
        return RWKV_ERROR_RUNTIME;
    }
    std::string path(model_path);
    std::string backend(backend_name);
    std::string tokenizer(tokenizer_path);
    rt->start_load_model_async();
    std::thread load_thread([rt, path, backend, tokenizer, extra]() {
        int ret = rt->load_model(path, backend, tokenizer, extra);
        int result_code = (ret >= 0) ? RWKV_SUCCESS : ret;
        int model_id = (ret >= 0) ? ret : -1;
        rt->set_load_model_result(result_code, model_id);
    });
    load_thread.detach();
    return RWKV_SUCCESS;
}

int rwkvmobile_runtime_is_loading_model(rwkvmobile_runtime_t runtime) {
    if (runtime == nullptr) {
        return 0;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    return rt->is_loading_model() ? 1 : 0;
}

void rwkvmobile_runtime_get_load_model_status(rwkvmobile_runtime_t runtime, int * result_code, int * model_id) {
    if (runtime == nullptr || result_code == nullptr || model_id == nullptr) {
        return;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    if (rt->is_loading_model()) {
        return;
    }
    int code = 0, id = -1;
    rt->get_load_model_result(code, id);
    *result_code = code;
    *model_id = id;
}

float rwkvmobile_runtime_get_load_model_progress(rwkvmobile_runtime_t runtime) {
    if (runtime == nullptr) {
        return -1.f;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    return rt->get_load_model_progress();
}

int rwkvmobile_runtime_release_model(rwkvmobile_runtime_t handle, int model_id) {
    if (handle == nullptr || model_id < 0) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class Runtime *>(handle);
    return rt->release_model(model_id);
}

int rwkvmobile_runtime_eval_logits(rwkvmobile_runtime_t handle, int model_id, const int * ids, int ids_len, float * logits, int logits_len) {
    if (handle == nullptr || ids == nullptr || logits == nullptr || ids_len <= 0 || logits_len <= 0) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class Runtime *>(handle);
    std::vector<int> ids_vec(ids, ids + ids_len);
    Tensor1D logits_ret;
    auto ret = rt->eval_logits(model_id, ids_vec, logits_ret);
    if (ret != RWKV_SUCCESS) {
        return ret;
    }
    const int n = std::min<int>(logits_len, (int)logits_ret.count);
    if (n <= 0 || logits_ret.data_ptr == nullptr) {
        return RWKV_ERROR_RUNTIME;
    }
    if (logits_ret.dtype == TensorDType::F32) {
        memcpy(logits, logits_ret.data_ptr, (size_t)n * sizeof(float));
    } else if (logits_ret.dtype == TensorDType::F16) {
        const half_float::half* h = reinterpret_cast<const half_float::half*>(logits_ret.data_ptr);
        for (int i = 0; i < n; ++i) {
            logits[i] = (float)h[i];
        }
    } else {
        return RWKV_ERROR_UNSUPPORTED;
    }
    return RWKV_SUCCESS;
}

int rwkvmobile_runtime_eval_chat_with_history_async(
    rwkvmobile_runtime_t handle,
    int model_id,
    const char ** inputs,
    const int num_inputs,
    const int max_tokens,
    void (*callback)(const char *, const int, const char *),
    int enable_reasoning,
    int force_reasoning,
    int force_lang,
    int add_generation_prompt
) {
    if (handle == nullptr || inputs == nullptr || num_inputs == 0 || max_tokens <= 0) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }

    auto rt = static_cast<class Runtime *>(handle);
    rt->set_is_generating(model_id, true);
    rt->set_stop_signal(model_id, false);
    std::vector<std::string> inputs_vec;
    for (int i = 0; i < num_inputs; i++) {
        inputs_vec.push_back(std::string(inputs[i]));
    }

    std::thread generation_thread([=]() {
        int ret = rt->chat(
            model_id,
            inputs_vec,
            max_tokens,
            callback,
            enable_reasoning != 0,
            force_reasoning != 0,
            add_generation_prompt != 0,
            force_lang,
            {});
        return ret;
    });

    generation_thread.detach();

    return RWKV_SUCCESS;
}

int rwkvmobile_runtime_eval_chat_batch_with_history_async(
    rwkvmobile_runtime_t handle,
    int model_id,
    const char *** inputs,
    const int * num_inputs,
    const int batch_size,
    const int max_tokens,
    void (*callback_batch)(const int, const char **, const int*, const char **),
    int enable_reasoning,
    int force_reasoning,
    int force_lang,
    int add_generation_prompt
) {
    if (handle == nullptr || inputs == nullptr || num_inputs == 0 || max_tokens <= 0 || batch_size <= 0) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }

    auto rt = static_cast<class Runtime *>(handle);
    rt->set_is_generating(model_id, true);
    rt->set_stop_signal(model_id, false);
    std::vector<std::vector<std::string>> inputs_vec(batch_size);
    for (int i = 0; i < batch_size; i++) {
        inputs_vec[i].resize(num_inputs[i]);
        for (int j = 0; j < num_inputs[i]; j++) {
            inputs_vec[i][j] = std::string(inputs[i][j]);
        }
    }

    std::thread generation_thread([=]() {
        int ret = rt->chat_batch(
            model_id,
            inputs_vec,
            max_tokens,
            batch_size,
            callback_batch,
            enable_reasoning != 0,
            force_reasoning != 0,
            add_generation_prompt != 0,
            force_lang,
            {});
        return ret;
    });

    generation_thread.detach();

    return RWKV_SUCCESS;
}

struct supported_batch_sizes rwkvmobile_runtime_get_supported_batch_sizes(rwkvmobile_runtime_t runtime, int model_id) {
    struct supported_batch_sizes sizes;
    sizes.sizes = nullptr;
    sizes.length = 0;
    if (runtime == nullptr) {
        return sizes;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    auto supported_batch_sizes = rt->get_supported_batch_sizes(model_id);
    sizes.length = supported_batch_sizes.size();
    sizes.sizes = (int *)malloc(sizes.length * sizeof(int));
    for (int i = 0; i < sizes.length; i++) {
        sizes.sizes[i] = supported_batch_sizes[i];
    }
    return sizes;
}

void rwkvmobile_runtime_free_supported_batch_sizes(struct supported_batch_sizes sizes) {
    if (sizes.sizes == nullptr) {
        return;
    }
    free(sizes.sizes);
}

int rwkvmobile_runtime_gen_completion_async(
    rwkvmobile_runtime_t handle,
    int model_id,
    const char * prompt,
    const int max_tokens,
    const int stop_code,
    void (*callback)(const char *, const int, const char *),
    int disable_cache) {
    if (handle == nullptr || prompt == nullptr || max_tokens <= 0) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }

    auto rt = static_cast<class Runtime *>(handle);
    rt->clear_response_buffer(model_id);
    rt->set_is_generating(model_id, true);
    rt->set_stop_signal(model_id, false);
    std::thread generation_thread([=]() {
        int ret = rt->gen_completion(
            model_id,
            std::string(prompt),
            max_tokens,
            stop_code,
            callback,
            disable_cache);
        return ret;
    });

    generation_thread.detach();

    return RWKV_SUCCESS;
}

const char ** rwkvmobile_runtime_gen_completion_singletoken_topk(
    rwkvmobile_runtime_t handle,
    int model_id,
    const char * prompt,
    const int top_k
) {
    if (handle == nullptr || prompt == nullptr || top_k <= 0) {
        return nullptr;
    }

    auto rt = static_cast<class Runtime *>(handle);
    static std::vector<std::string> candidate_output_texts;
    static std::vector<const char *> candidate_output_texts_list(top_k, nullptr);
    int ret = rt->gen_completion_singletoken_topk(model_id, std::string(prompt), top_k, candidate_output_texts, nullptr);
    if (ret != RWKV_SUCCESS) {
        return nullptr;
    }
    if (candidate_output_texts.size() != top_k) {
        LOGE("gen_completion_singletoken_topk: candidate_output_texts.size() != top_k");
        return nullptr;
    }

    for (int i = 0; i < candidate_output_texts.size(); i++) {
        candidate_output_texts_list[i] = candidate_output_texts[i].c_str();
    }
    return candidate_output_texts_list.data();
}

int rwkvmobile_runtime_gen_completion_batch_async(
    rwkvmobile_runtime_t handle,
    int model_id,
    const char ** prompts,
    const int batch_size,
    const int max_tokens,
    const int stop_code,
    void (*callback_batch)(const int, const char **, const int*, const char **),
    int disable_cache) {
    if (handle == nullptr || prompts == nullptr || batch_size <= 0 || max_tokens <= 0) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }

    auto rt = static_cast<class Runtime *>(handle);
    rt->clear_response_buffer(model_id);
    rt->set_is_generating(model_id, true);
    rt->set_stop_signal(model_id, false);

    std::thread generation_thread([=]() {
        std::vector<std::string> prompts_vec(batch_size);
        for (int i = 0; i < batch_size; i++) {
            prompts_vec[i] = std::string(prompts[i]);
        }
        int ret = rt->gen_completion_batch(
            model_id,
            prompts_vec,
            batch_size,
            max_tokens,
            stop_code,
            callback_batch,
            disable_cache);
        return ret;
    });

    generation_thread.detach();

    return RWKV_SUCCESS;
}

int rwkvmobile_runtime_gen_completion(
    rwkvmobile_runtime_t handle,
    int model_id,
    const char * prompt,
    const int max_tokens,
    const int stop_code,
    void (*callback)(const char *, const int, const char *),
    int disable_cache) {
    if (handle == nullptr || prompt == nullptr || max_tokens <= 0 || callback == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }

    auto rt = static_cast<class Runtime *>(handle);
    rt->clear_response_buffer(model_id);
    return rt->gen_completion(
        model_id,
        std::string(prompt),
        max_tokens,
        stop_code,
        callback,
        disable_cache);
}

int rwkvmobile_runtime_stop_generation(rwkvmobile_runtime_t runtime, int model_id) {
    if (runtime == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    rt->set_stop_signal(model_id, true);
    return RWKV_SUCCESS;
}

int rwkvmobile_runtime_is_generating(rwkvmobile_runtime_t runtime, int model_id) {
    if (runtime == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    bool is_generating = rt->is_generating(model_id);
    return is_generating;
}

int rwkvmobile_runtime_clear_state(rwkvmobile_runtime_t handle, int model_id) {
    if (handle == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class Runtime *>(handle);
    return rt->clear_state(model_id);
}

int rwkvmobile_runtime_load_initial_state(rwkvmobile_runtime_t handle, int model_id, const char * state_path) {
    if (handle == nullptr || state_path == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class Runtime *>(handle);
    return rt->load_initial_state(model_id, state_path);
}

void rwkvmobile_runtime_unload_initial_state(rwkvmobile_runtime_t handle, int model_id, const char * state_path) {
    if (handle == nullptr) {
        return;
    }
    auto rt = static_cast<class Runtime *>(handle);
    rt->unload_initial_state(model_id, state_path);
}

int rwkvmobile_runtime_save_history_to_state(
    rwkvmobile_runtime_t handle,
    int model_id,
    const char ** history,
    const int num_history,
    const char * state_path) {
    if (handle == nullptr || history == nullptr || num_history <= 0 || state_path == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }

    std::vector<std::string> inputs_vec;
    for (int i = 0; i < num_history; i++) {
        inputs_vec.push_back(std::string(history[i]));
    }
    auto rt = static_cast<class Runtime *>(handle);
    return rt->save_state_by_history(model_id, inputs_vec, state_path);
}

int rwkvmobile_runtime_load_history_state_to_memory(rwkvmobile_runtime_t handle, int model_id, const char * state_path) {
    if (handle == nullptr || state_path == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class Runtime *>(handle);
    return rt->load_history_state_to_memory(model_id, state_path);
}

int rwkvmobile_runtime_get_available_backend_names(char * backend_names_buffer, int buffer_size) {
    if (backend_names_buffer == nullptr || buffer_size <= 0) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    Runtime * rt = new Runtime();
    if (rt == nullptr) {
        return RWKV_ERROR_ALLOC;
    }
    auto backend_names = rt->get_available_backends_str();
    if (backend_names.size() >= buffer_size) {
        return RWKV_ERROR_ALLOC;
    }
    strncpy(backend_names_buffer, backend_names.c_str(), buffer_size);
    delete rt;
    return RWKV_SUCCESS;
}

struct sampler_params rwkvmobile_runtime_get_sampler_params(rwkvmobile_runtime_t runtime, int model_id) {
    struct sampler_params params;
    params.temperature = 0;
    params.top_k = 0;
    params.top_p = 0;
    if (runtime == nullptr) {
        return params;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    params.temperature = rt->get_temperature(model_id);
    params.top_k = rt->get_top_k(model_id);
    params.top_p = rt->get_top_p(model_id);
    return params;
}

struct sampler_params rwkvmobile_runtime_get_sampler_params_on_batch_slot(rwkvmobile_runtime_t runtime, int model_id, int slot) {
    struct sampler_params params;
    params.temperature = 0;
    params.top_k = 0;
    params.top_p = 0;
    if (runtime == nullptr) {
        return params;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    params.temperature = rt->get_temperature_on_batch_slot(model_id, slot);
    params.top_k = rt->get_top_k_on_batch_slot(model_id, slot);
    params.top_p = rt->get_top_p_on_batch_slot(model_id, slot);
    return params;
}

void rwkvmobile_runtime_set_sampler_params(rwkvmobile_runtime_t runtime, int model_id, struct sampler_params params) {
    if (runtime == nullptr) {
        return;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    rt->set_sampler_params(model_id, params.temperature, params.top_k, params.top_p);
}

void rwkvmobile_runtime_set_sampler_params_on_batch_slot(rwkvmobile_runtime_t runtime, int model_id, int slot, struct sampler_params params) {
    if (runtime == nullptr) {
        return;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    rt->set_sampler_params_on_batch_slot(model_id, slot, params.temperature, params.top_k, params.top_p);
}

struct penalty_params rwkvmobile_runtime_get_penalty_params(rwkvmobile_runtime_t runtime, int model_id) {
    struct penalty_params params;
    params.presence_penalty = 0;
    params.frequency_penalty = 0;
    params.penalty_decay = 0;
    if (runtime == nullptr) {
        return params;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    params.presence_penalty = rt->get_presence_penalty(model_id);
    params.frequency_penalty = rt->get_frequency_penalty(model_id);
    params.penalty_decay = rt->get_penalty_decay(model_id);
    return params;
}

struct penalty_params rwkvmobile_runtime_get_penalty_params_on_batch_slot(rwkvmobile_runtime_t runtime, int model_id, int slot) {
    struct penalty_params params;
    params.presence_penalty = 0;
    params.frequency_penalty = 0;
    params.penalty_decay = 0;
    if (runtime == nullptr) {
        return params;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    params.presence_penalty = rt->get_presence_penalty_on_batch_slot(model_id, slot);
    params.frequency_penalty = rt->get_frequency_penalty_on_batch_slot(model_id, slot);
    params.penalty_decay = rt->get_penalty_decay_on_batch_slot(model_id, slot);
    return params;
}

void rwkvmobile_runtime_set_penalty_params(rwkvmobile_runtime_t runtime, int model_id, struct penalty_params params) {
    if (runtime == nullptr) {
        return;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    rt->set_penalty_params(model_id, params.presence_penalty, params.frequency_penalty, params.penalty_decay);
}

void rwkvmobile_runtime_set_penalty_params_on_batch_slot(rwkvmobile_runtime_t runtime, int model_id, int slot, struct penalty_params params) {
    if (runtime == nullptr) {
        return;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    rt->set_penalty_params_on_batch_slot(model_id, slot, params.presence_penalty, params.frequency_penalty, params.penalty_decay);
}

int rwkvmobile_runtime_set_prompt(rwkvmobile_runtime_t runtime, int model_id, const char * prompt) {
    if (runtime == nullptr || prompt == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    return rt->set_prompt(model_id, prompt);
}

int rwkvmobile_runtime_get_prompt(rwkvmobile_runtime_t runtime, int model_id, char * prompt, const int buf_len) {
    if (runtime == nullptr || prompt == nullptr || buf_len <= 0) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    std::string prompt_str = rt->get_prompt(model_id);
    if (prompt_str.size() >= buf_len) {
        return RWKV_ERROR_ALLOC;
    }
    strncpy(prompt, prompt_str.c_str(), buf_len);
    return RWKV_SUCCESS;
}

void rwkvmobile_runtime_add_adsp_library_path(const char * path) {
#ifndef _WIN32
    auto ld_lib_path_char = getenv("LD_LIBRARY_PATH");
    std::string ld_lib_path;
    if (ld_lib_path_char) {
        ld_lib_path = std::string(path) + ":" + std::string(ld_lib_path_char);
    } else {
        ld_lib_path = std::string(path);
    }
    LOGI("Setting LD_LIBRARY_PATH to %s\n", ld_lib_path.c_str());
    setenv("LD_LIBRARY_PATH", ld_lib_path.c_str(), 1);
    setenv("ADSP_LIBRARY_PATH", path, 1);
#endif
}

void rwkvmobile_runtime_set_qnn_library_path(rwkvmobile_runtime_t runtime, const char * path) {
#ifndef _WIN32
    auto ld_lib_path_char = getenv("LD_LIBRARY_PATH");
    std::string ld_lib_path;
    if (ld_lib_path_char) {
        ld_lib_path = std::string(path) + ":" + std::string(ld_lib_path_char);
    } else {
        ld_lib_path = std::string(path);
    }
    LOGI("Setting LD_LIBRARY_PATH to %s\n", ld_lib_path.c_str());
    setenv("LD_LIBRARY_PATH", ld_lib_path.c_str(), 1);
    setenv("ADSP_LIBRARY_PATH", path, 1);
#endif
}

double rwkvmobile_runtime_get_avg_decode_speed(rwkvmobile_runtime_t runtime, int model_id) {
    if (runtime == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    return rt->get_avg_decode_speed(model_id);
}

double rwkvmobile_runtime_get_avg_prefill_speed(rwkvmobile_runtime_t runtime, int model_id) {
    if (runtime == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    return rt->get_avg_prefill_speed(model_id);
}

int rwkvmobile_runtime_load_vision_encoder(rwkvmobile_runtime_t runtime, int model_id, const char * encoder_path) {
#if ENABLE_VISION
    if (runtime == nullptr || encoder_path == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    return rt->load_vision_encoder(model_id, encoder_path);
#else
    return RWKV_ERROR_UNSUPPORTED;
#endif
}

int rwkvmobile_runtime_load_vision_encoder_and_adapter(rwkvmobile_runtime_t runtime, int model_id, const char * encoder_path, const char * adapter_path) {
#if ENABLE_VISION
    if (runtime == nullptr || encoder_path == nullptr || adapter_path == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    return rt->load_vision_encoder(model_id, encoder_path, adapter_path);
#else
    return RWKV_ERROR_UNSUPPORTED;
#endif
}

int rwkvmobile_runtime_release_vision_encoder(rwkvmobile_runtime_t runtime, int model_id) {
#if ENABLE_VISION
    if (runtime == nullptr) {
        return RWKV_SUCCESS;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    return rt->release_vision_encoder(model_id);
#else
    return RWKV_ERROR_UNSUPPORTED;
#endif
}

int rwkvmobile_runtime_set_image_unique_identifier(rwkvmobile_runtime_t runtime, const char * unique_identifier) {
    if (runtime == nullptr || unique_identifier == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    return rt->set_image_unique_identifier(unique_identifier);
}

int rwkvmobile_runtime_load_whisper_encoder(rwkvmobile_runtime_t runtime, int model_id, const char * encoder_path) {
#if ENABLE_WHISPER
    if (runtime == nullptr || encoder_path == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    return rt->load_whisper_encoder(model_id, encoder_path);
#else
    return RWKV_ERROR_UNSUPPORTED;
#endif
}

int rwkvmobile_runtime_release_whisper_encoder(rwkvmobile_runtime_t runtime, int model_id) {
#if ENABLE_WHISPER
    if (runtime == nullptr) {
        return RWKV_SUCCESS;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    return rt->release_whisper_encoder(model_id);
#else
    return RWKV_ERROR_UNSUPPORTED;
#endif
}

int rwkvmobile_runtime_set_audio_prompt(rwkvmobile_runtime_t runtime, int model_id, const char * audio_path) {
#if ENABLE_WHISPER
    if (runtime == nullptr || audio_path == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    return rt->set_audio_prompt(model_id, audio_path);
#else
    return RWKV_ERROR_UNSUPPORTED;
#endif
}

int rwkvmobile_runtime_set_token_banned(rwkvmobile_runtime_t runtime, int model_id, const int * token_banned, int token_banned_len) {
    if (runtime == nullptr || token_banned == nullptr || token_banned_len <= 0) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    std::vector<int> token_banned_vec(token_banned, token_banned + token_banned_len);
    rt->set_token_banned(model_id, token_banned_vec);
    return RWKV_SUCCESS;
}

int rwkvmobile_runtime_set_eos_token(rwkvmobile_runtime_t runtime, int model_id, const char * eos_token) {
    if (runtime == nullptr || eos_token == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    rt->set_eos_token(model_id, eos_token);
    return RWKV_SUCCESS;
}

int rwkvmobile_runtime_set_bos_token(rwkvmobile_runtime_t runtime, int model_id, const char * bos_token) {
    if (runtime == nullptr || bos_token == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    rt->set_bos_token(model_id, bos_token);
    return RWKV_SUCCESS;
}

int rwkvmobile_runtime_set_user_role(rwkvmobile_runtime_t runtime, int model_id, const char * user_role) {
    if (runtime == nullptr || user_role == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    rt->set_user_role(model_id, user_role);
    return RWKV_SUCCESS;
}

int rwkvmobile_runtime_set_response_role(rwkvmobile_runtime_t runtime, int model_id, const char * response_role) {
    if (runtime == nullptr || response_role == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    rt->set_response_role(model_id, response_role);
    return RWKV_SUCCESS;
}

int rwkvmobile_runtime_set_thinking_token(rwkvmobile_runtime_t runtime, int model_id, const char * thinking_token) {
    if (runtime == nullptr || thinking_token == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    rt->set_thinking_token(model_id, thinking_token);
    return RWKV_SUCCESS;
}

int rwkvmobile_runtime_set_space_after_roles(rwkvmobile_runtime_t runtime, int model_id, int space_after_roles) {
    if (runtime == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    rt->set_space_after_roles(model_id, (bool)space_after_roles);
    return RWKV_SUCCESS;
}

struct response_buffer rwkvmobile_runtime_get_response_buffer_content(rwkvmobile_runtime_t runtime, int model_id) {
    struct response_buffer buffer;
    buffer.content = nullptr;
    buffer.length = 0;
    buffer.eos_found = false;
    if (runtime == nullptr) {
        return buffer;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    std::string content = rt->get_response_buffer_content(model_id);
    buffer.length = content.size();
    buffer.content = (char *)malloc(buffer.length * sizeof(char));
    if (buffer.content == nullptr) {
        return buffer;
    }
    memset(buffer.content, 0, buffer.length);
    strncpy(buffer.content, content.c_str(), buffer.length);
    buffer.eos_found = rt->get_response_buffer_eos_found(model_id);
    return buffer;
}

int rwkvmobile_runtime_get_response_buffer_tokens_count(rwkvmobile_runtime_t runtime, int model_id) {
    if (runtime == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    return rt->get_response_buffer_tokens_count(model_id);
}

struct batch_tokens_count rwkvmobile_runtime_get_response_buffer_tokens_count_batch(rwkvmobile_runtime_t runtime, int model_id) {
    if (runtime == nullptr) {
        return {nullptr, 0};
    }
    auto rt = static_cast<class Runtime *>(runtime);
    std::vector<int> counts = rt->get_response_buffer_tokens_count_batch(model_id);
    static int token_counts_static[32];
    if (counts.size() > 32) {
        return {nullptr, 0};
    }

    struct batch_tokens_count count;
    count.counts = token_counts_static;
    count.batch_size = counts.size();
    for (int i = 0; i < counts.size(); i++) {
        count.counts[i] = counts[i];
    }
    return count;
}

int rwkvmobile_runtime_calculate_tokens_count_from_messages(rwkvmobile_runtime_t runtime, int model_id, const char ** inputs, const int num_inputs) {
    if (runtime == nullptr || inputs == nullptr || num_inputs <= 0) {
        return 0;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    std::vector<std::string> inputs_vec;
    for (int i = 0; i < num_inputs; i++) {
        inputs_vec.push_back(std::string(inputs[i]));
    }
    return rt->calculate_tokens_count_from_messages(model_id, inputs_vec);
}

int rwkvmobile_runtime_calculate_tokens_count_from_text(rwkvmobile_runtime_t runtime, int model_id, const char * text) {
    if (runtime == nullptr || text == nullptr) {
        return 0;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    return rt->calculate_tokens_count_from_text(model_id, std::string(text));
}

void rwkvmobile_runtime_free_response_buffer(struct response_buffer buffer) {
    if (buffer.content == nullptr) {
        return;
    }
    free((void *)buffer.content);
}

struct response_buffer_batch rwkvmobile_runtime_get_response_buffer_content_batch(rwkvmobile_runtime_t runtime, int model_id) {
    struct response_buffer_batch buffer;
    buffer.contents = nullptr;
    buffer.lengths = nullptr;
    buffer.eos_founds = nullptr;
    buffer.batch_size = 0;
    if (runtime == nullptr) {
        return buffer;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    std::vector<std::string> contents = rt->get_response_buffer_content_batch(model_id);
    auto eos_founds = rt->get_response_buffer_eos_found_batch(model_id);
    buffer.batch_size = contents.size();
    buffer.contents = (char **)malloc(contents.size() * sizeof(char *));
    buffer.lengths = (int *)malloc(contents.size() * sizeof(int));
    buffer.eos_founds = (int *)malloc(contents.size() * sizeof(int));
    for (int i = 0; i < contents.size(); i++) {
        buffer.contents[i] = (char *)malloc(contents[i].size() * sizeof(char));
        strncpy(buffer.contents[i], contents[i].c_str(), contents[i].size());
        buffer.lengths[i] = contents[i].size();
        buffer.eos_founds[i] = eos_founds[i];
    }
    return buffer;
}

void rwkvmobile_runtime_free_response_buffer_batch(struct response_buffer_batch buffer) {
    if (buffer.contents == nullptr) {
        return;
    }   
    for (int i = 0; i < buffer.batch_size; i++) {
        if (buffer.contents[i] != nullptr)
            free(buffer.contents[i]);
    }
    if (buffer.lengths != nullptr)
        free(buffer.lengths);
    if (buffer.eos_founds != nullptr)
        free(buffer.eos_founds);
    free(buffer.contents);
}

struct token_ids rwkvmobile_runtime_get_response_buffer_ids(rwkvmobile_runtime_t runtime, int model_id) {
    struct token_ids ids;
    ids.ids = nullptr;
    ids.len = 0;
    if (runtime == nullptr) {
        return ids;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    auto ids_vec = rt->get_response_buffer_ids(model_id);
    ids.ids = (int32_t *)malloc(ids_vec.size() * sizeof(int32_t));
    if (ids.ids == nullptr) {
        return ids;
    }
    for (int i = 0; i < ids_vec.size(); i++) {
        ids.ids[i] = ids_vec[i];
    }
    ids.len = ids_vec.size();
    return ids;
}

void rwkvmobile_runtime_free_token_ids(struct token_ids ids) {
    if (ids.ids == nullptr) {
        return;
    }
    free(ids.ids);
}

int rwkvmobile_runtime_sparktts_load_models(rwkvmobile_runtime_t runtime, const char * wav2vec2_path, const char * bicodec_tokenizer_path, const char * bicodec_detokenizer_path) {
#if ENABLE_TTS
    if (runtime == nullptr || wav2vec2_path == nullptr || bicodec_tokenizer_path == nullptr || bicodec_detokenizer_path == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    return rt->sparktts_load_models(wav2vec2_path, bicodec_tokenizer_path, bicodec_detokenizer_path);
#else
    return RWKV_ERROR_UNSUPPORTED;
#endif
}

int rwkvmobile_runtime_sparktts_release_models(rwkvmobile_runtime_t runtime) {
#if ENABLE_TTS
    if (runtime == nullptr) {
        return RWKV_SUCCESS;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    return rt->sparktts_release_models();
#else
    return RWKV_SUCCESS;
#endif
}

int rwkvmobile_runtime_run_spark_tts_streaming_async(rwkvmobile_runtime_t runtime, int model_id, const char * tts_text, const char * prompt_audio_text, const char * prompt_audio_path, const char * output_wav_path) {
#if ENABLE_TTS
    if (runtime == nullptr || tts_text == nullptr || prompt_audio_path == nullptr || output_wav_path == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    rt->tts_clear_streaming_buffer();
    rt->set_is_generating(model_id, true);
    rt->set_stop_signal(model_id, false);
    std::string prompt_audio_text_str;
    if (prompt_audio_text == nullptr) {
        prompt_audio_text_str = "";
    } else {
        prompt_audio_text_str = std::string(prompt_audio_text);
    }
    std::thread generation_thread([=]() {
        int ret = rt->run_spark_tts_zeroshot_streaming(model_id, tts_text, prompt_audio_text_str, prompt_audio_path, output_wav_path);
        return ret;
    });

    generation_thread.detach();
    return RWKV_SUCCESS;
#else
    return RWKV_ERROR_UNSUPPORTED;
#endif
}

int rwkvmobile_runtime_run_spark_tts_with_global_tokens_streaming_async(rwkvmobile_runtime_t runtime, int model_id, const char * tts_text, const char * output_wav_path, const int * global_tokens) {
#if ENABLE_TTS
    if (runtime == nullptr || tts_text == nullptr || output_wav_path == nullptr || global_tokens == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    rt->tts_clear_streaming_buffer();
    rt->set_is_generating(model_id, true);
    rt->set_stop_signal(model_id, false);
    std::vector<int> global_tokens_vec(global_tokens, global_tokens + 32);
    std::thread generation_thread([=]() {
        int ret = rt->run_spark_tts_with_global_tokens_streaming(model_id, tts_text, output_wav_path, global_tokens_vec);
        return ret;
    });

    generation_thread.detach();
    return RWKV_SUCCESS;
#else
    return RWKV_ERROR_UNSUPPORTED;
#endif
}

int rwkvmobile_runtime_run_spark_tts_with_properties_streaming_async(rwkvmobile_runtime_t runtime, int model_id, const char * tts_text, const char * output_wav_path, const char * age, const char * gender, const char * emotion, const char * pitch, const char * speed) {
#if ENABLE_TTS
    if (runtime == nullptr || tts_text == nullptr || output_wav_path == nullptr || age == nullptr || gender == nullptr || emotion == nullptr || pitch == nullptr || speed == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    rt->tts_clear_streaming_buffer();
    rt->set_is_generating(model_id, true);
    rt->set_stop_signal(model_id, false);
    std::thread generation_thread([=]() {
        int ret = rt->run_spark_tts_with_properties_streaming(model_id, tts_text, output_wav_path, age, gender, emotion, pitch, speed);
        return ret;
    });

    generation_thread.detach();
    return RWKV_SUCCESS;
#else
    return RWKV_ERROR_UNSUPPORTED;
#endif
}

struct tts_streaming_buffer rwkvmobile_runtime_get_tts_streaming_buffer(rwkvmobile_runtime_t runtime) {
    struct tts_streaming_buffer ret;
#if ENABLE_TTS
    auto rt = static_cast<class Runtime *>(runtime);
    std::lock_guard<std::mutex> lock(rt->_tts_streaming_buffer_mutex);
    auto buffer = rt->tts_get_streaming_buffer();
    ret.samples = new float[buffer.size()];
    memcpy(ret.samples, buffer.data(), buffer.size() * sizeof(float));
    ret.length = buffer.size();
#else
    ret.samples = nullptr;
    ret.length = 0;
#endif
    return ret;
}

int rwkvmobile_runtime_get_tts_streaming_buffer_length(rwkvmobile_runtime_t runtime) {
#if ENABLE_TTS
    if (runtime == nullptr) {
        return 0;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    return rt->tts_get_streaming_buffer().size();
#else
    return 0;
#endif
}

void rwkvmobile_runtime_free_tts_streaming_buffer(struct tts_streaming_buffer buffer) {
    if (buffer.samples == nullptr) {
        return;
    }
    delete[] buffer.samples;
}

const int * rwkvmobile_runtime_get_tts_global_tokens_output(rwkvmobile_runtime_t runtime) {
#if ENABLE_TTS
    if (runtime == nullptr) {
        return nullptr;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    return rt->tts_get_global_tokens_output().data();
#else
    return nullptr;
#endif
}

int rwkvmobile_runtime_tts_register_text_normalizer(rwkvmobile_runtime_t runtime, const char * path) {
#if ENABLE_TTS
    if (runtime == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    return rt->tts_register_text_normalizer(std::string(path));
#else
    return RWKV_ERROR_UNSUPPORTED;
#endif
}

float rwkvmobile_runtime_get_prefill_progress(rwkvmobile_runtime_t runtime, int model_id) {
    if (runtime == nullptr) {
        return 0;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    return rt->get_prefill_progress(model_id);
}

const char * rwkvmobile_get_platform_name() {
    soc_detect soc_detect;
    soc_detect.detect_platform();
    return soc_detect.get_platform_name();
}

const char * rwkvmobile_get_soc_name() {
    soc_detect soc_detect;
    soc_detect.detect_platform();
    return soc_detect.get_soc_name();
}

const char * rwkvmobile_get_soc_partname() {
    soc_detect soc_detect;
    soc_detect.detect_platform();
    return soc_detect.get_soc_partname();
}

const char * rwkvmobile_get_htp_arch() {
    soc_detect soc_detect;
    soc_detect.detect_platform();
    return soc_detect.get_htp_arch();
}

const char * rwkvmobile_dump_log() {
    return logger_get_log().c_str();
}

const char * rwkvmobile_get_state_cache_info(rwkvmobile_runtime_t runtime, int model_id) {
    if (runtime == nullptr) {
        return nullptr;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    auto state_cache_info = rt->get_state_cache_info(model_id);
    char * state_cache_info_str = (char *)malloc(state_cache_info.size() + 1);
    strcpy(state_cache_info_str, state_cache_info.c_str());
    return (const char *)state_cache_info_str;
}

void rwkvmobile_free_state_cache_info(const char * state_cache_info) {
    if (state_cache_info == nullptr) {
        return;
    }
    free((void *)state_cache_info);
}

void rwkvmobile_set_loglevel(int loglevel) {
    logger_set_loglevel(loglevel);
}

void rwkvmobile_set_cache_dir(rwkvmobile_runtime_t runtime, const char * cache_dir) {
    if (cache_dir == nullptr) {
        return;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    rt->set_cache_dir(std::string(cache_dir));
}

int rwkvmobile_load_embedding_model(rwkvmobile_runtime_t runtime, const char *model_path) {
#ifdef ENABLE_LLAMACPP
    if (runtime == nullptr || model_path == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    return rt->load_embedding_model(model_path);
#else
    return RWKV_ERROR_UNSUPPORTED;
#endif
}

int rwkvmobile_load_rerank_model(rwkvmobile_runtime_t runtime, const char *model_path) {
#ifdef ENABLE_LLAMACPP
    if (runtime == nullptr || model_path == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    return rt->load_rerank_model(model_path);
#else
    return RWKV_ERROR_UNSUPPORTED;
#endif
}

int rwkvmobile_get_embedding(rwkvmobile_runtime_t runtime, const char **input, const int input_length, float **embedding) {
#ifdef ENABLE_LLAMACPP
    if (runtime == nullptr || input == nullptr || embedding == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    std::vector<std::string> inputs;
    for (int i = 0; i < input_length; i++) {
        inputs.emplace_back(input[i]);
    }

    auto ebd = rt->get_embedding(inputs);
    if (ebd.empty()) {
        return RWKV_ERROR_EVAL;
    }
    memcpy(embedding, ebd.data(), ebd[0].size() * ebd.size() * sizeof(float));
    return 0;
#else
    return RWKV_ERROR_UNSUPPORTED;
#endif
}

int rwkvmobile_runtime_get_loaded_model_ids(rwkvmobile_runtime_t handle, int * model_ids, int max_count) {
    if (handle == nullptr || model_ids == nullptr || max_count <= 0) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }

    auto rt = static_cast<class Runtime *>(handle);
    auto loaded_ids = rt->get_loaded_model_ids();

    int count = std::min(static_cast<int>(loaded_ids.size()), max_count);
    for (int i = 0; i < count; i++) {
        model_ids[i] = loaded_ids[i];
    }

    return count;
}

struct loaded_models_list rwkvmobile_runtime_get_loaded_models_info(rwkvmobile_runtime_t handle) {
    struct loaded_models_list result = {nullptr, 0};

    if (handle == nullptr) {
        return result;
    }

    auto rt = static_cast<class Runtime *>(handle);
    auto models_info = rt->get_loaded_models_info();

    if (models_info.empty()) {
        return result;
    }

    result.count = models_info.size();
    result.models = static_cast<struct model_info*>(malloc(sizeof(struct model_info) * result.count));

    if (result.models == nullptr) {
        result.count = 0;
        return result;
    }

    int index = 0;
    for (const auto& pair : models_info) {
        const auto& info = pair.second;
        struct model_info* model = &result.models[index];

        model->model_id = pair.first;

        auto allocate_string = [](const std::string& str) -> char* {
            char* result = static_cast<char*>(malloc(str.length() + 1));
            if (result) {
                strcpy(result, str.c_str());
            }
            return result;
        };

        model->model_path = allocate_string(info.at("model_path"));
        model->backend_name = allocate_string(info.at("backend_name"));
        model->tokenizer_path = allocate_string(info.at("tokenizer_path"));
        model->user_role = allocate_string(info.at("user_role"));
        model->response_role = allocate_string(info.at("response_role"));
        model->bos_token = allocate_string(info.at("bos_token"));
        model->eos_token = allocate_string(info.at("eos_token"));
        model->thinking_token = allocate_string(info.at("thinking_token"));
        model->is_generating = (info.at("is_generating") == "true") ? 1 : 0;
        model->vocab_size = std::stoi(info.at("vocab_size"));

        index++;
    }

    return result;
}

void rwkvmobile_runtime_free_loaded_models_list(struct loaded_models_list list) {
    if (list.models == nullptr || list.count == 0) {
        return;
    }

    for (int i = 0; i < list.count; i++) {
        struct model_info* model = &list.models[i];

        free(model->model_path);
        free(model->backend_name);
        free(model->tokenizer_path);
        free(model->user_role);
        free(model->response_role);
        free(model->bos_token);
        free(model->eos_token);
        free(model->thinking_token);
    }

    free(list.models);
}

const char * rwkvmobile_runtime_get_model_path_by_id(rwkvmobile_runtime_t runtime, int model_id) {
    if (runtime == nullptr) {
        return nullptr;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    return rt->get_model_path_by_id(model_id).c_str();
}


struct evaluation_results rwkvmobile_runtime_run_evaluation(rwkvmobile_runtime_t runtime, int model_id, const char * source_text, const char * target_text) {
    struct evaluation_results result;
    if (runtime == nullptr || source_text == nullptr || target_text == nullptr) {
        return result;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    bool correct = false;
    float logits_val = 0;
    std::string output_text;
    int ret = rt->run_evaluation(model_id, source_text, target_text, correct, logits_val, output_text, true);
    if (ret != RWKV_SUCCESS) {
        return result;
    }

    result.count = 1;
    result.corrects = new int[1];
    result.corrects[0] = correct;
    result.logits_vals = new float[1];
    result.logits_vals[0] = logits_val;
    result.output_texts = new char*[1];
    result.output_texts[0] = new char[output_text.size() + 1];
    strcpy(result.output_texts[0], output_text.c_str());
    return result;
}

void rwkvmobile_runtime_free_evaluation_results(struct evaluation_results results) {
    if (results.corrects != nullptr) {
        delete[] results.corrects;
    }
    if (results.logits_vals != nullptr) {
        delete[] results.logits_vals;
    }
    if (results.output_texts != nullptr) {
        for (int i = 0; i < results.count; i++) {
            if (results.output_texts[i] != nullptr) {
                delete[] results.output_texts[i];
            }
        }
        delete[] results.output_texts;
    }
}

int rwkvmobile_runtime_set_seed(rwkvmobile_runtime_t runtime, int model_id, int seed) {
    if (runtime == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    return rt->set_seed(model_id, seed);
}

int rwkvmobile_runtime_get_seed(rwkvmobile_runtime_t runtime, int model_id) {
    if (runtime == nullptr) {
        return 0;
    }
    auto rt = static_cast<class Runtime *>(runtime);
    return rt->get_seed(model_id);
}

int rwkvmobile_convert_pth_to_safetensors(const char * pth_path, const char * st_path) {
#ifdef ENABLE_WEBRWKV
    if (pth_path == nullptr || st_path == nullptr) {
        LOGE("Invalid parameters: pth_path: %s, st_path: %s\n", pth_path == nullptr ? "nullptr" : pth_path, st_path == nullptr ? "nullptr" : st_path);
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    return ::convert_pth_to_st(pth_path, st_path);
#else
    LOGE("WebRWKV backend is not enabled on this platform\n");
    return RWKV_ERROR_UNSUPPORTED;
#endif
}

} // extern "C"
} // namespace rwkvmobile
