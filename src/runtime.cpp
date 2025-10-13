#include "runtime.h"
#include "backend.h"
#include "logger.h"
#include <functional>
#include <filesystem>
#include <chrono>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <thread>
#include "rmpack.h"
#include "utils.h"
#ifdef ENABLE_WEBRWKV
#include "web_rwkv_backend.h"
#endif

#ifdef ENABLE_NCNN
#include "ncnn_rwkv_backend.h"
#endif

#ifdef ENABLE_LLAMACPP
#include "llama_cpp_backend.h"
#endif

#ifdef ENABLE_QNN
#include "qnn_backend.h"
#endif

#ifdef ENABLE_MNN
#include "mnn_rwkv_backend.h"
#endif

#ifdef ENABLE_COREML
#include "coreml_rwkv_backend.h"
#endif

#if defined(ENABLE_VISION)
#include "multimodal/vision/vision_encoder.h"
#endif

#if defined(ENABLE_WHISPER)
#include "multimodal/whisper/whisper_encoder.h"
#endif

#if defined(ENABLE_TTS)
#include "frontend_utils.h"
#include "tts_properties.h"
#endif

#if defined(ENABLE_TTS) || defined(ENABLE_WHISPER)
#include "audio.h"
#endif

namespace rwkvmobile {

std::string backend_enum_to_str(int backend) {
    switch (backend) {
        case RWKV_BACKEND_WEBRWKV:
            return "web-rwkv";
        case RWKV_BACKEND_NCNN:
            return "ncnn";
        case RWKV_BACKEND_LLAMACPP:
            return "llama.cpp";
        case RWKV_BACKEND_QNN:
            return "qnn";
        case RWKV_BACKEND_MNN:
            return "mnn";
        case RWKV_BACKEND_COREML:
            return "coreml";
        default:
            return "unknown";
    }
}

int backend_str_to_enum(std::string backend) {
    if (backend == "web-rwkv") {
        return RWKV_BACKEND_WEBRWKV;
    } else if (backend == "ncnn") {
        return RWKV_BACKEND_NCNN;
    } else if (backend == "llama.cpp") {
        return RWKV_BACKEND_LLAMACPP;
    } else if (backend == "qnn") {
        return RWKV_BACKEND_QNN;
    } else if (backend == "mnn") {
        return RWKV_BACKEND_MNN;
    } else if (backend == "coreml") {
        return RWKV_BACKEND_COREML;
    }
    return -1;
}

int runtime::load_model(std::string model_path, std::string backend_name, std::string tokenizer_path, void * extra) {
    int ret_model_id = -1;
    int backend_id = backend_str_to_enum(backend_name);
    if (backend_id < 0) {
        LOGE("Invalid backend name: %s\n", backend_name.c_str());
        return ret_model_id;
    }

    auto model_instance = std::make_unique<ModelInstance>();
    if (model_instance == nullptr) {
        LOGE("Failed to allocate memory for model instance\n");
        return ret_model_id;
    }

    // 1. Create and initialize backend
    if (backend_id == RWKV_BACKEND_WEBRWKV) {
#ifdef ENABLE_WEBRWKV
        model_instance->backend = std::unique_ptr<execution_provider, std::function<void(execution_provider*)>>(new web_rwkv_backend,
            [](execution_provider *p) { delete (web_rwkv_backend*)p; });
#else
        LOGE("WebRWKV backend is not supported on this platform\n");
        return ret_model_id;
#endif
    } else if (backend_id == RWKV_BACKEND_NCNN) {
#ifdef ENABLE_NCNN
        model_instance->backend = std::unique_ptr<execution_provider, std::function<void(execution_provider*)>>(new ncnn_rwkv_backend,
            [](execution_provider *p) { delete (ncnn_rwkv_backend*)p; });
#else
        LOGE("NCNN backend is not supported on this platform\n");
        return ret_model_id;
#endif
    } else if (backend_id == RWKV_BACKEND_LLAMACPP) {
#ifdef ENABLE_LLAMACPP
        model_instance->backend = std::unique_ptr<execution_provider, std::function<void(execution_provider*)>>(new llama_cpp_backend,
            [](execution_provider *p) { delete (llama_cpp_backend*)p; });
#else
        LOGE("LLaMa.cpp backend is not supported on this platform\n");
        return ret_model_id;
#endif
    } else if (backend_id == RWKV_BACKEND_QNN) {
#ifdef ENABLE_QNN
        model_instance->backend = std::unique_ptr<execution_provider, std::function<void(execution_provider*)>>(new qnn_backend,
            [](execution_provider *p) { delete (qnn_backend*)p; });
#else
        LOGE("QNN backend is not supported on this platform\n");
        return ret_model_id;
#endif
    } else if (backend_id == RWKV_BACKEND_MNN) {
#ifdef ENABLE_MNN
        model_instance->backend = std::unique_ptr<execution_provider, std::function<void(execution_provider*)>>(new mnn_rwkv_backend,
            [](execution_provider *p) { delete (mnn_rwkv_backend*)p; });
#else
        LOGE("MNN backend is not supported on this platform\n");
        return ret_model_id;
#endif
    } else if (backend_id == RWKV_BACKEND_COREML) {
#ifdef ENABLE_COREML
        model_instance->backend = std::unique_ptr<execution_provider, std::function<void(execution_provider*)>>(new coreml_rwkv_backend,
            [](execution_provider *p) { delete (coreml_rwkv_backend*)p; });
#else
        LOGE("CoreML backend is not supported on this platform\n");
        return ret_model_id;
#endif
    } else {
        LOGE("Unsupported backend: %s\n", backend_name.c_str());
        return ret_model_id;
    }

    int ret = model_instance->backend->init(extra);
    if (ret) {
        LOGE("Failed to initialize backend: %s, errno = %d\n", backend_name.c_str(), ret);
        return ret_model_id;
    }

    // 2. Load model
    ret = model_instance->backend->load_model(model_path);
    if (ret) {
        LOGE("Failed to load model from: %s, errno = %d\n", model_path.c_str(), ret);
        return ret_model_id;
    }
    LOGI("Loaded model from: %s as model_id = %d\n", model_path.c_str(), _next_model_id);
    LOGI("Model num_layers: %d, num_heads: %d, hidden_size: %d, vocab_size: %d\n",
         model_instance->backend->n_layers, model_instance->backend->num_heads, model_instance->backend->hidden_size, model_instance->backend->vocab_size);
    model_instance->backend->zero_state();
    model_instance->backend->state_root = std::make_unique<state_node>();
    if (model_instance->backend->state_root == nullptr) {
        return ret_model_id;
    }

    model_instance->backend->state_root->activation_count = 20;
    model_instance->backend->get_state(model_instance->backend->state_root->state);
    model_instance->backend->state_root->ids = std::vector<int>();
    model_instance->backend->state_root->logits = std::vector<float>(model_instance->backend->vocab_size, 0);

    // 3. Load tokenizer
    if (!tokenizer_path.empty()) {
        model_instance->tokenizer = std::unique_ptr<tokenizer_base, std::function<void(tokenizer_base*)>>(new trie_tokenizer,
            [](tokenizer_base *p) { delete (trie_tokenizer*)p; });
        if (model_instance->tokenizer == nullptr) {
            return ret_model_id;
        }
        ret = model_instance->tokenizer->load(tokenizer_path);
        if (ret) {
            LOGE("[LOAD_MODEL] Failed to load tokenizer for model ID %d", _next_model_id);
            return ret_model_id;
        }
    }

    // 4. Create sampler
    model_instance->sampler = std::make_unique<NucleusSampler>();
    if (model_instance->sampler == nullptr) {
        return ret_model_id;
    }

    // 5. Store the instance and return its ID
    ret_model_id = _next_model_id++;
    _models[ret_model_id] = std::move(model_instance);
    _models[ret_model_id]->model_path = model_path;
    _models[ret_model_id]->backend_name = backend_name;
    _models[ret_model_id]->tokenizer_path = tokenizer_path;

    return ret_model_id;
}

int runtime::release_model(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    _models.erase(model_id);
    return RWKV_SUCCESS;
}

#ifdef ENABLE_VISION
int runtime::load_vision_encoder(int model_id, std::string model_path, std::string adapter_path) {
    if (_models.find(model_id) == _models.end()) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);
    model->multimodal_encoder = std::make_unique<VisionEncoder>();
    if (model->multimodal_encoder == nullptr) {
        return RWKV_ERROR_ALLOC;
    }
    return model->multimodal_encoder->LoadModel(model_path, adapter_path);
}

int runtime::release_vision_encoder(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);
    model->multimodal_encoder.reset();
    return RWKV_SUCCESS;
}
#endif

int runtime::set_image_unique_identifier(std::string unique_identifier) {
    _image_unique_identifier = unique_identifier;
    return RWKV_SUCCESS;
}

#ifdef ENABLE_WHISPER
int runtime::load_whisper_encoder(int model_id, std::string model_path) {
    if (_models.find(model_id) == _models.end()) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);
    model->multimodal_encoder = std::make_unique<WhisperEncoder>();
    if (model->multimodal_encoder == nullptr) {
        return RWKV_ERROR_ALLOC;
    }
    return model->multimodal_encoder->LoadModel(model_path, "");
}

int runtime::release_whisper_encoder(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);
    model->multimodal_encoder.reset();
    return RWKV_SUCCESS;
}
#endif

int runtime::get_available_backend_ids(std::vector<int> &backend_ids) {
    backend_ids = std::vector<int>();

#ifdef ENABLE_WEBRWKV
    // Snapdragon platform doesn't support WebRWKV backend yet
    if (_soc_detect.get_platform_type() != PLATFORM_SNAPDRAGON) {
        backend_ids.push_back(RWKV_BACKEND_WEBRWKV);
    }
#endif

#ifdef ENABLE_NCNN
    backend_ids.push_back(RWKV_BACKEND_NCNN);
#endif

#ifdef ENABLE_LLAMACPP
    backend_ids.push_back(RWKV_BACKEND_LLAMACPP);
#endif

#ifdef ENABLE_QNN
    if (_soc_detect.get_platform_type() == PLATFORM_SNAPDRAGON) {
        auto supported_soc_names = std::vector<std::string>{"SM8650", "SM8635", "SM8550", "SM8475"};
        if (std::find(supported_soc_names.begin(), supported_soc_names.end(), _soc_detect.get_soc_partname()) != supported_soc_names.end()) {
            backend_ids.push_back(RWKV_BACKEND_QNN);
        }
    }
#endif

#ifdef ENABLE_COREML
    backend_ids.push_back(RWKV_BACKEND_COREML);
#endif

    return RWKV_SUCCESS;
}

std::string runtime::get_available_backends_str() {
    std::vector<int> backend_ids;
    get_available_backend_ids(backend_ids);
    std::string ret = "";
    for (auto id : backend_ids) {
        ret += backend_enum_to_str(id) + ",";
    }
    return ret;
}

int runtime::load_initial_state(int model_id, std::string state_path) {
    if (_models.find(model_id) == _models.end()) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);
    if (state_path.find(".rmpack") == std::string::npos) {
        LOGE("the specified state file is not a rmpack file\n");
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    std::string initial_state_str = "<state src=\"" + state_path + "\">";
    std::vector<int> initial_state_ids = model->tokenizer->encode(initial_state_str);

    for (int i = 0; i < model->backend->state_root->children.size(); i++) {
        if (model->backend->state_root->children[i]->ids == initial_state_ids) {
            return RWKV_SUCCESS;
        }
    }

    std::any initial_state;
    RMPackReader state_pack(state_path);
    int hidden_size_config = state_pack.getConfig()["hidden_size"];
    auto files = state_pack.getFiles();
    if (files.size() != model->backend->n_layers) {
        LOGE("state file has %d layers, but model has %d layers\n", (int)files.size(), model->backend->n_layers);
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    if (hidden_size_config != model->backend->hidden_size) {
        LOGE("state file has hidden size %d, but model has hidden size %d\n", hidden_size_config, model->backend->hidden_size);
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    size_t state_size = state_pack.getFileSize(files[0].filename);

    std::vector<std::vector<half_float::half>> states(model->backend->n_layers);
    for (int i = 0; i < model->backend->n_layers; i++) {
        states[i].resize(state_size / sizeof(half_float::half));
        auto data = state_pack.readFileToMemory(files[i].filename);
        memcpy(states[i].data(), data, state_size);
        state_pack.freeFileMemory(files[i].filename);
    }

    model->backend->load_raw_states(states);
    model->backend->get_state(initial_state);
    // make a new constant state node
    model->backend->state_root->children.push_back(std::make_unique<state_node>(initial_state, initial_state_ids, std::vector<float>(model->backend->vocab_size, 0), true));
    model->backend->state_root->children.back()->activation_count = 50;
    return RWKV_SUCCESS;
}

void runtime::unload_initial_state(int model_id, std::string state_path) {
    if (_models.find(model_id) == _models.end()) {
        return;
    }
    auto &model = _models.at(model_id);
    std::string initial_state_str = "<state src=\"" + state_path + "\">";
    std::vector<int> initial_state_ids = model->tokenizer->encode(initial_state_str);
    for (int i = 0; i < model->backend->state_root->children.size(); i++) {
        if (model->backend->state_root->children[i]->ids == initial_state_ids) {
            model->backend->state_root->children.erase(model->backend->state_root->children.begin() + i);
            break;
        }
    }
}

int runtime::eval_logits(int model_id, int id, float *& logits) {
    if (_models.find(model_id) == _models.end()) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);
    if (model->backend == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto start = std::chrono::high_resolution_clock::now();
    int ret = model->backend->eval(id, logits);
    auto end = std::chrono::high_resolution_clock::now();
    _decode_speed = 1e6f / std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    return ret;
}

int runtime::eval_logits(int model_id, std::vector<int> ids, float *& logits) {
    if (_models.find(model_id) == _models.end()) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);
    if (model->backend == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    auto start = std::chrono::high_resolution_clock::now();
    int i = 0;
    int ret;
    for (; i + _prefill_chunk_size <= ids.size(); i += _prefill_chunk_size) {
        auto ids_chunk = std::vector<int>(ids.begin() + i, ids.begin() + i + _prefill_chunk_size);
        ret = model->backend->eval(ids_chunk, logits);
        if (ret != RWKV_SUCCESS) return ret;
        if (_current_prefill_total_tokens > 0) {
            _current_prefill_finished_tokens += _prefill_chunk_size;
            _prefill_progress = (double)_current_prefill_finished_tokens / _current_prefill_total_tokens;
            LOGD("Update prefill_progress = %f", _prefill_progress);
        }
    }
    if (i < ids.size()) {
        auto ids_left = std::vector<int>(ids.begin() + i, ids.end());
        ret = model->backend->eval(ids_left, logits);
        if (_current_prefill_total_tokens > 0) {
            _current_prefill_finished_tokens += ids_left.size();
            _prefill_progress = (double)_current_prefill_finished_tokens / _current_prefill_total_tokens;
            LOGD("Update prefill_progress = %f", _prefill_progress);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    _prefill_speed = ids.size() * 1e6f / std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    return ret;
}

int runtime::eval_logits_with_embeddings(int model_id, const float *embeddings, int n_tokens, float *& logits) {
    if (_models.find(model_id) == _models.end()) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);
    if (model->backend == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto start = std::chrono::high_resolution_clock::now();
    auto ret = model->backend->eval_with_embeddings(embeddings, n_tokens, logits);
    auto end = std::chrono::high_resolution_clock::now();
    if (n_tokens > 1) {
        _prefill_speed = n_tokens * 1e6f / std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    } else {
        _decode_speed = 1e6f / std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
    return ret;
}

int runtime::eval_logits_batch_decode(int model_id, std::vector<int> ids, float *& logits) {
    if (_models.find(model_id) == _models.end()) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);
    if (model->backend == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<int>> ids_batch(ids.size());
    for (int i = 0; i < ids.size(); i++) {
        ids_batch[i] = std::vector<int>(1);
        ids_batch[i][0] = ids[i];
    }

    int ret = model->backend->eval_batch(ids_batch, logits);
    auto end = std::chrono::high_resolution_clock::now();
    _decode_speed = ids.size() * 1e6f / std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    return ret;
}

std::vector<int> runtime::get_supported_batch_sizes(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return {};
    }
    auto &model = _models.at(model_id);
    return model->backend->supported_batch_sizes;
}

std::string runtime::apply_chat_template(int model_id, std::vector<std::string> inputs, bool enable_reasoning) {
    if (_models.find(model_id) == _models.end()) {
        return "";
    }
    auto &model = _models.at(model_id);

    static auto replace_text = [](const std::string& text, const std::string& old_str, const std::string& new_str) -> std::string {
        std::string result = text;
        size_t pos = 0;
        while ((pos = result.find(old_str, pos)) != std::string::npos) {
            result.replace(pos, old_str.length(), new_str);
            pos += new_str.length();
        }
        return result;
    };

    std::string text = model->prompt;
    for (int i = 0; i < inputs.size(); i++) {
        if (i % 2 == 0) {
            auto user_text = inputs[i];
            user_text = replace_text(user_text, "\r\n", "\n");
            user_text = replace_text(user_text, "\n\n", "\n");

            text += model->bos_token + model->user_role + ": " + inputs[i] + model->eos_token;
        } else {
            if (i == inputs.size() - 1) {
                text += model->bos_token + model->response_role + ": " + inputs[i];
            } else {
                text += model->bos_token + model->response_role + ": " + inputs[i] + model->eos_token;
            }
        }
    }

    if (inputs.size() % 2 != 0) {
        text +=  model->response_role + ":";
        if (enable_reasoning) {
            text += " " + model->thinking_token;
        }
    }
    return text;
}

std::vector<runtime::TokenChunk> runtime::split_text_by_image_and_token_num(const std::string text, int max_tokens_per_chunk, int model_id) {
    std::vector<TokenChunk> chunks;
    std::string remaining_text = text;

    // Helper: split by <image> tags as before
    while (!remaining_text.empty()) {
        std::string image_unique_identifier_start = "<" + _image_unique_identifier + ">";
        std::string image_unique_identifier_end = "</" + _image_unique_identifier + ">";
        size_t image_start = remaining_text.find(image_unique_identifier_start);
        if (image_start != std::string::npos) {
            size_t image_end = remaining_text.find(image_unique_identifier_end, image_start);
            if (image_end != std::string::npos) {
                // Add text before image tag as a regular chunk if not empty
                if (image_start > 0) {
                    std::string before_image = remaining_text.substr(0, image_start);
                    if (!before_image.empty()) {
                        auto tokens = _models.at(model_id)->tokenizer->encode(before_image);
                        chunks.push_back({tokens, false, ""});
                    }
                }

                // Extract image path and encode the full image tag as tokens
                size_t path_start = image_start + image_unique_identifier_start.length();
                std::string image_path = remaining_text.substr(path_start, image_end - path_start);
                std::string full_image_tag = remaining_text.substr(image_start, image_end + image_unique_identifier_end.length() - image_start); // include both tags
                auto image_tokens = _models.at(model_id)->tokenizer->encode(full_image_tag);
                chunks.push_back({image_tokens, true, image_path});

                // Update remaining text
                remaining_text = remaining_text.substr(image_end + image_unique_identifier_end.length());
                continue;
            }
        }

        // No more image tags found, process remaining text
        if (!remaining_text.empty()) {
            auto tokens = _models.at(model_id)->tokenizer->encode(remaining_text);
            chunks.push_back({tokens, false, ""});
            break;
        }
    }

    // Now split regular token chunks by max_tokens_per_chunk, also split at "Assistant"
    std::vector<TokenChunk> final_chunks;
    const std::string assistant_str = "Assistant";
    for (const auto& chunk : chunks) {
        if (chunk.is_image) {
            final_chunks.push_back(chunk);
        } else {
            // Decode tokens to string for splitting at "Assistant"
            const std::vector<int>& tokens_to_split = chunk.tokens;
            if (tokens_to_split.empty()) continue;

            // Decode to text
            std::string chunk_text = _models.at(model_id)->tokenizer->decode(tokens_to_split);

            size_t pos = 0;
            size_t last_pos = 0;
            std::vector<std::string> split_texts;
            std::vector<bool> assistant_belongs;
            while ((pos = chunk_text.find(assistant_str, last_pos)) != std::string::npos) {
                // If "Assistant" is at the start, skip empty chunk
                if (pos > last_pos) {
                    split_texts.push_back(chunk_text.substr(last_pos, pos - last_pos));
                    assistant_belongs.push_back(false);
                }
                // "Assistant" itself as a chunk (belongs to next chunk)
                split_texts.push_back(assistant_str);
                assistant_belongs.push_back(true);
                last_pos = pos + assistant_str.length();
            }
            // Add the remaining part
            if (last_pos < chunk_text.size()) {
                split_texts.push_back(chunk_text.substr(last_pos));
                assistant_belongs.push_back(false);
            }

            // Now, for each split_text, encode and split by max_tokens_per_chunk
            std::vector<int> carry_tokens;
            for (size_t i = 0; i < split_texts.size(); ++i) {
                std::string part = split_texts[i];
                if (assistant_belongs[i]) {
                    // This is "Assistant", append to carry_tokens for next chunk
                    auto assistant_tokens = _models.at(model_id)->tokenizer->encode(part);
                    carry_tokens.insert(carry_tokens.end(), assistant_tokens.begin(), assistant_tokens.end());
                    continue;
                }
                // Normal part
                auto part_tokens = _models.at(model_id)->tokenizer->encode(part);
                // Prepend any carried "Assistant" tokens
                if (!carry_tokens.empty()) {
                    part_tokens.insert(part_tokens.begin(), carry_tokens.begin(), carry_tokens.end());
                    carry_tokens.clear();
                }
                // Split by max_tokens_per_chunk
                for (size_t j = 0; j < part_tokens.size(); j += max_tokens_per_chunk) {
                    size_t end = std::min(j + max_tokens_per_chunk, part_tokens.size());
                    std::vector<int> token_chunk(part_tokens.begin() + j, part_tokens.begin() + end);
                    if (!token_chunk.empty()) {
                        final_chunks.push_back({token_chunk, false, ""});
                    }
                }
            }
            // If any "Assistant" tokens left, put them as a chunk
            if (!carry_tokens.empty()) {
                final_chunks.push_back({carry_tokens, false, ""});
            }
        }
    }

    return final_chunks;
}

int runtime::save_state_by_history(int model_id, std::vector<std::string> history, std::string state_path) {
    if (_models.find(model_id) == _models.end()) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);
    if (model->backend == nullptr || model->tokenizer == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    if (history.size() % 2 != 0) {
        history.pop_back();
    }

    auto input_text = apply_chat_template(model_id, history, false);
    std::vector<int> text_ids = model->tokenizer->encode(input_text);

    std::vector<int> tokens_to_prefill;
    auto node = model->backend->match_and_load_state(text_ids, tokens_to_prefill);
    auto matched_ids = node->ids;

    LOGI("saving state cache for model \"%s\" and backend: \"%s\" for prefix: \"%s\" ", model->model_path.c_str(), model->backend_name.c_str(), escape_special_chars(model->tokenizer->decode(matched_ids)).c_str());

    std::vector<uint8_t> state_data;
    int ret = model->backend->serialize_runtime_state(node->state, state_data);
    if (ret) {
        LOGE("failed to serialize runtime state\n");
        return ret;
    }

    // Prepare ids data
    const auto& ids_data = matched_ids;
    uint32_t state_size = static_cast<uint32_t>(state_data.size());
    uint32_t ids_size = static_cast<uint32_t>(ids_data.size() * sizeof(int));
    uint32_t logits_size = static_cast<uint32_t>(node->logits.size() * sizeof(float));

    // Write to file
    FILE* f = fopen(state_path.c_str(), "wb");
    if (!f) {
        LOGE("failed to open state_path for writing: %s\n", state_path.c_str());
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_IO;
    }

    size_t write_cnt = 0;
    // Write [state_size_in_bytes]
    write_cnt = fwrite(&state_size, sizeof(uint32_t), 1, f);
    if (write_cnt != 1) {
        LOGE("failed to write state size\n");
        fclose(f);
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_IO;
    }
    // Write [state_data]
    if (!state_data.empty()) {
        write_cnt = fwrite(state_data.data(), 1, state_data.size(), f);
        if (write_cnt != state_data.size()) {
            LOGE("failed to write state data\n");
            fclose(f);
            return RWKV_ERROR_RUNTIME | RWKV_ERROR_IO;
        }
    }
    // Write [ids_size_in_bytes]
    write_cnt = fwrite(&ids_size, sizeof(uint32_t), 1, f);
    if (write_cnt != 1) {
        LOGE("failed to write ids size\n");
        fclose(f);
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_IO;
    }
    // Write [ids_data]
    if (!ids_data.empty()) {
        write_cnt = fwrite(ids_data.data(), sizeof(int), ids_data.size(), f);
        if (write_cnt != ids_data.size()) {
            LOGE("failed to write ids data\n");
            fclose(f);
            return RWKV_ERROR_RUNTIME | RWKV_ERROR_IO;
        }
    }
    // Write [logits_size_in_bytes]
    write_cnt = fwrite(&logits_size, sizeof(uint32_t), 1, f);
    if (write_cnt != 1) {
        LOGE("failed to write logits size\n");
        fclose(f);
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_IO;
    }
    // Write [logits_data]
    if (!node->logits.empty()) {
        write_cnt = fwrite(node->logits.data(), sizeof(float), node->logits.size(), f);
        if (write_cnt != node->logits.size()) {
            LOGE("failed to write logits data\n");
            fclose(f);
            return RWKV_ERROR_RUNTIME | RWKV_ERROR_IO;
        }
    }

    fclose(f);

    LOGI("State and ids saved successfully to %s (state size: %u bytes, ids count: %zu, ids size: %u bytes)", state_path.c_str(), state_size, ids_data.size(), ids_size);

    return RWKV_SUCCESS;
}

int runtime::load_history_state_to_memory(int model_id, std::string state_path) {
    if (_models.find(model_id) == _models.end()) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);
    if (model->backend == nullptr || model->tokenizer == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    FILE* f = fopen(state_path.c_str(), "rb");
    if (!f) {
        LOGE("failed to open state_path for reading: %s\n", state_path.c_str());
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_IO;
    }

    uint32_t state_size;
    size_t read_cnt = fread(&state_size, sizeof(uint32_t), 1, f);
    if (read_cnt != 1) {
        LOGE("failed to read state size\n");
        fclose(f);
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_IO;
    }

    std::vector<uint8_t> state_data(state_size);
    read_cnt = fread(state_data.data(), 1, state_size, f);
    if (read_cnt != state_size) {
        LOGE("failed to read state data\n");
        fclose(f);
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_IO;
    }

    uint32_t ids_size;
    read_cnt = fread(&ids_size, sizeof(uint32_t), 1, f);
    if (read_cnt != 1) {
        LOGE("failed to read ids size\n");
        fclose(f);
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_IO;
    }

    std::vector<int> ids_data(ids_size / sizeof(int));
    read_cnt = fread(ids_data.data(), sizeof(int), ids_size / sizeof(int), f);
    if (read_cnt != ids_size / sizeof(int)) {
        LOGE("failed to read ids data\n");
        fclose(f);
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_IO;
    }

    uint32_t logits_size;
    read_cnt = fread(&logits_size, sizeof(uint32_t), 1, f);
    if (read_cnt != 1) {
        LOGE("failed to read logits size\n");
        fclose(f);
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_IO;
    }
    std::vector<float> logits_data(logits_size / sizeof(float));
    read_cnt = fread(logits_data.data(), sizeof(float), logits_size / sizeof(float), f);
    if (read_cnt != logits_size / sizeof(float)) {
        LOGE("failed to read logits data\n");
        fclose(f);
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_IO;
    }

    fclose(f);
    LOGI("loaded state_size: %u bytes, ids_size: %u bytes, logits_size: %u bytes", state_size, ids_size, logits_size);

    std::any runtime_state;
    model->backend->deserialize_runtime_state(state_data, runtime_state);

    std::vector<int> tokens_to_prefill;
    auto node = model->backend->match_and_load_state(ids_data, tokens_to_prefill);
    model->backend->register_state_checkpoint_with_state(node, ids_data, logits_data.data(), runtime_state);
    LOGI("loaded state from disk for text: \"%s\"", escape_special_chars(model->tokenizer->decode(node->ids)).c_str());

    return RWKV_SUCCESS;
}

std::string runtime::get_state_cache_info(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return "";
    }
    auto &model = _models.at(model_id);
    auto state_root = model->backend->state_root.get();

    std::string state_cache_info;
    std::vector<state_node*> tmp;
    tmp.push_back(state_root);
    // traverse the state tree
    while (!tmp.empty()) {
        auto node = tmp.back();
        tmp.pop_back();
        state_cache_info += "text = \"" + escape_special_chars(model->tokenizer->decode(node->ids)) + "\", remaining lifespan = " + std::to_string(node->activation_count) + "\n";
        for (auto &child : node->children) {
            tmp.push_back(child.get());
        }
    }
    return state_cache_info;
}

int runtime::chat(int model_id, std::vector<std::string> inputs, const int max_length, void (*callback)(const char *, const int, const char *), bool enable_reasoning) {
    if (_models.find(model_id) == _models.end()) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);
    if (model->backend == nullptr || model->tokenizer == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    model->is_generating = true;
    model->stop_signal = false;
    model->response_buffer.clear();
    model->response_buffer_ids.clear();
    model->response_buffer_eos_found = false;

    if (_prefilling_thread.joinable() && _prefilling_thread.get_id() != std::this_thread::get_id()) {
        LOGD("Found prefilling thread, joining\n");
        _prefilling_thread.join();
        LOGD("_prefilling_thread finished.\n");
    }

    auto input_text = apply_chat_template(model_id, inputs, enable_reasoning);
    LOGD("Applied chat template: \"%s\"\n", input_text.c_str());
    std::vector<int> text_ids = model->tokenizer->encode(input_text);
    std::string debug_msg = "text_ids: ";
    for (auto id : text_ids) {
        debug_msg += std::to_string(id) + " ";
    }
    LOGD("%s\n", debug_msg.c_str());

    float *logits = nullptr;
    std::vector<int> tokens_to_prefill;
    auto node = model->backend->match_and_load_state(text_ids, tokens_to_prefill);
    LOGI("matched state cache for prefix: \"%s\"", escape_special_chars(model->tokenizer->decode(node->ids)).c_str());

    if (tokens_to_prefill.size() > 0) {
        _prefill_progress_start(tokens_to_prefill.size());
        auto text_to_prefill = model->tokenizer->decode(tokens_to_prefill);
        LOGI("new text to prefill: \"%s\"", escape_special_chars(text_to_prefill).c_str());

        // Split text by image tags and token count, then process each chunk
        int checkpoint_interval = 256;
        auto token_chunks = split_text_by_image_and_token_num(text_to_prefill, checkpoint_interval, model_id);

        for (const auto& chunk : token_chunks) {
            if (chunk.is_image) {
                // Handle image chunk - encode the image tag for state matching
                LOGD("Processing image chunk: %s\n", chunk.image_path.c_str());
                if (!chunk.tokens.empty()) {
                    int ret;
#ifdef ENABLE_VISION
                    // "<|vision_start|>" = 65530
                    // "<|vision_end|>" = 65531
                    ret = eval_logits(model_id, 65530, logits);
                    if (ret) {
                        model->is_generating = false;
                        LOGE("failed to eval logits for image chunk\n");
                        return ret;
                    }
                    auto start = std::chrono::high_resolution_clock::now();
                    std::vector<float> embeddings;
                    int n_tokens;
                    if (!model->multimodal_encoder->Encode(chunk.image_path, embeddings, n_tokens, model->backend->embedding_input_force_no_ln0())) {
                        model->is_generating = false;
                        LOGE("failed to encode image for image chunk\n");
                        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
                    }
                    auto end = std::chrono::high_resolution_clock::now();
                    LOGI("siglip duration: %lld ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
                    ret = eval_logits_with_embeddings(model_id, embeddings.data(), n_tokens, logits);
                    if (ret) {
                        model->is_generating = false;
                        LOGE("failed to eval logits with embeddings for image chunk\n");
                        return ret;
                    }
                    ret = eval_logits(model_id, 65531, logits);
                    if (ret) {
                        model->is_generating = false;
                        LOGE("failed to eval logits for image chunk\n");
                        return ret;
                    }
#endif
                    ret = model->backend->register_state_checkpoint(node, chunk.tokens, logits);
                    if (ret) {
                        model->is_generating = false;
                        LOGE("failed to register state checkpoint for image chunk\n");
                        return ret;
                    }
                    LOGI("registered state for text: \"%s\"", escape_special_chars(model->tokenizer->decode(node->ids)).c_str());
                }
            } else {
                if (chunk.tokens.empty()) {
                    continue;
                }

                int ret = eval_logits(model_id, chunk.tokens, logits);
                if (ret) {
                    model->is_generating = false;
                    LOGE("failed to eval logits\n");
                    return ret;
                }
                ret = model->backend->register_state_checkpoint(node, chunk.tokens, logits);
                if (ret) {
                    model->is_generating = false;
                    LOGE("failed to register state checkpoint\n");
                    return ret;
                }
                LOGI("registered state for text: \"%s\"", escape_special_chars(model->tokenizer->decode(node->ids)).c_str());
            }
        }
    }
    _prefill_progress_finish();

    model->response_buffer = input_text.substr(input_text.rfind(model->response_role + ":") + (model->response_role + ":").size());
    std::vector<int> response_ids_raw;
    model->response_buffer_ids = model->tokenizer->encode(model->response_buffer);
    int ret;

    model->sampler->clear_occurences();
    if (inputs.size() % 2 == 0) {
        std::vector<int> ids = model->tokenizer->encode(" " + inputs[inputs.size() - 1]);
        for (auto id: ids) {
            model->sampler->update_occurences(id);
        }
    }

    if (logits == nullptr) {
        if (node->logits.size() == model->backend->get_num_vocab()) {
            logits = node->logits.data();
        } else {
            LOGE("no logits found, neither from saved state nor from new tokens to prefill\n");
            ret = eval_logits(model_id, text_ids.back(), logits);
            if (ret) {
                model->is_generating = false;
                LOGE("failed to eval logits\n");
                return ret;
            }
            response_ids_raw.emplace_back(text_ids.back());
        }
    }

    int decoded_idx = 0;
    bool thinking_end_tag_found = false;
    bool is_pseudo_thinking = enable_reasoning && model->response_buffer.find("</think>") != std::string::npos;
    const int rewind_token_list[] = {28324, 28329, 10080, 9830}; // "…\n" "。\n" "…" "。"
    std::any state_for_rewinding;

    for (int i = 0; i < max_length; i++) {
        model->sampler->apply_penalties(logits, model->backend->get_num_vocab());

        if (i == 0) {
            if (is_pseudo_thinking) {
                // token 61 is '<', 261 is '\n\n'
                logits[61] = -1e9f;
                logits[261] = -1e9f;
            }
            logits[0] = -1e9f;
        } else if (is_pseudo_thinking && i == 1 && decoded_idx == 11) {
            logits[61] = -1e9f;
        }

        decoded_idx = model->sampler->sample(logits, model->backend->get_num_vocab());
        if (decoded_idx == 0) {
            LOGD("sampled token 0, stopping generation\n");
            break;
        }

        std::string decoded = model->tokenizer->decode(decoded_idx);
        std::string tmp = model->response_buffer + decoded;
        for (auto &stop_code : model->stop_codes) {
            if (enable_reasoning && !thinking_end_tag_found && stop_code == "\n\n") {
                continue;
            }
            if (tmp.size() >= stop_code.size() &&
                tmp.compare(tmp.size() - stop_code.size(), stop_code.size(), stop_code) == 0) {
                LOGD("stop code found: %s\n", stop_code.c_str());
                model->response_buffer_eos_found = true;
                break;
            }
        }

        if (enable_reasoning && !thinking_end_tag_found) {
            if (tmp.find("</think>") != std::string::npos) {
                thinking_end_tag_found = true;
            }
        }

        if (model->response_buffer_eos_found || model->stop_signal) {
            LOGD("stopping generation, eos_found: %d, stop_signal: %d\n", model->response_buffer_eos_found, model->stop_signal);
            break;
        } else if (state_for_rewinding.has_value()) {
            state_for_rewinding.reset();
        }

        if (std::any_of(rewind_token_list, rewind_token_list + sizeof(rewind_token_list) / sizeof(rewind_token_list[0]), [decoded_idx](int token) { return decoded_idx == token; })) {
            model->backend->get_state(state_for_rewinding);
        }

        ret = eval_logits(model_id, decoded_idx, logits);
        if (ret) {
            model->is_generating = false;
            LOGE("failed to eval logits\n");
            return ret;
        }

        response_ids_raw.emplace_back(decoded_idx);
        model->response_buffer += decoded;
        model->response_buffer_ids.emplace_back(decoded_idx);
        if (i == 0 && model->response_buffer[0] == ' ') {
            model->response_buffer = model->response_buffer.substr(1);
        }

        model->sampler->update_occurences(decoded_idx);
        if (callback) {
            callback(model->response_buffer.c_str(), decoded_idx, decoded.c_str());
        }
    }

    if (response_ids_raw.size() > 0) {
        int ret;
        if (state_for_rewinding.has_value()) {
            response_ids_raw.pop_back();
            ret = model->backend->register_state_checkpoint_with_state(node, response_ids_raw, logits, state_for_rewinding);
        } else {
            ret = model->backend->register_state_checkpoint(node, response_ids_raw, logits);
        }
        if (ret) {
            model->is_generating = false;
            LOGE("failed to register state checkpoint\n");
            return ret;
        }
        LOGI("registered state for text: \"%s\"", escape_special_chars(model->tokenizer->decode(node->ids)).c_str());
    }

    model->is_generating = false;
    model->stop_signal = false;

    model->backend->cleanup_state_tree();

    return RWKV_SUCCESS;
}

int runtime::chat_batch(int model_id, std::vector<std::vector<std::string>> inputs, const int max_length, const int batch_size, void (*callback_batch)(const int, const char **, const int*, const char **), bool enable_reasoning) {
    if (_models.find(model_id) == _models.end()) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);
    if (model->backend == nullptr || model->tokenizer == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    bool supported = false;
    for (auto size : model->backend->supported_batch_sizes) {
        if (batch_size == size) {
            supported = true;
            break;
        }
    }
    if (!supported) {
        LOGE("chat_batch: batch size %d is not supported\n", batch_size);
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_UNSUPPORTED;
    }

    if (inputs.size() != batch_size) {
        LOGE("chat_batch: inputs size %d is not equal to batch size %d\n", inputs.size(), batch_size);
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    model->is_generating = true;
    model->stop_signal = false;

    model->response_buffer_batch.resize(batch_size);
    model->response_buffer_ids_batch.resize(batch_size);
    model->response_buffer_eos_found_batch.resize(batch_size);

    std::vector<std::vector<int>> response_ids_raw_batch(batch_size);

    std::vector<std::string> input_texts(batch_size);
    std::vector<std::vector<int>> text_ids_batch(batch_size);
    // std::vector<float*> logits_batch(batch_size, nullptr);
    float *logits = nullptr;
    std::vector<state_node*> nodes_batch(batch_size);

    std::vector<std::map<int, float>> occurences_batch(batch_size);
    std::vector<int> decoded_idx(batch_size);
    std::vector<bool> is_pseudo_thinking_batch(batch_size, false);
    std::vector<std::any> state_batch(batch_size);
    std::vector<bool> thinking_end_tag_found_batch(batch_size, false);
    const int rewind_token_list[] = {28324, 28329, 10080, 9830}; // "…\n" "。\n" "…" "。"
    std::vector<std::any> state_for_rewinding_batch(batch_size);

    int ret;
    auto num_vocab = model->backend->get_num_vocab();

    // prefill for each batch
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        auto &input = inputs[batch_idx];
        input_texts[batch_idx] = apply_chat_template(model_id, input, enable_reasoning);
        LOGD("Applied chat template for batch %d: \"%s\"\n", batch_idx, input_texts[batch_idx].c_str());
        text_ids_batch[batch_idx] = model->tokenizer->encode(input_texts[batch_idx]);

        model->response_buffer_batch[batch_idx] = input_texts[batch_idx].substr(input_texts[batch_idx].rfind(model->response_role + ":") + (model->response_role + ":").size());;
        model->response_buffer_ids_batch[batch_idx].clear();
        model->response_buffer_eos_found_batch[batch_idx] = false;
        std::vector<int> tokens_to_prefill;
        nodes_batch[batch_idx] = model->backend->match_and_load_state(text_ids_batch[batch_idx], tokens_to_prefill);
        LOGI("batch %d matched state cache for prefix: \"%s\"", batch_idx, escape_special_chars(model->tokenizer->decode(nodes_batch[batch_idx]->ids)).c_str());

        // prefill needed tokens
        if (tokens_to_prefill.size() > 0) {
            _prefill_progress_start(tokens_to_prefill.size());
            LOGI("new text to prefill: \"%s\"", escape_special_chars(model->tokenizer->decode(tokens_to_prefill)).c_str());

            // save a state checkpoint every about 256 tokens
            int checkpoint_interval = 256;
            for (int j = 0; j < tokens_to_prefill.size(); j += checkpoint_interval) {
                std::vector<int> tokens_to_prefill_chunk = std::vector<int>(tokens_to_prefill.begin() + j, tokens_to_prefill.begin() + std::min(j + checkpoint_interval, (int)tokens_to_prefill.size()));
                int ret = eval_logits(model_id, tokens_to_prefill_chunk, logits);
                if (ret) {
                    model->is_generating = false;
                    LOGE("failed to eval logits\n");
                    return ret;
                }
                ret = model->backend->register_state_checkpoint(nodes_batch[batch_idx], tokens_to_prefill_chunk, logits);
                if (ret) {
                    model->is_generating = false;
                    LOGE("failed to register state checkpoint\n");
                    return ret;
                }
                LOGI("registered state for text: \"%s\"", escape_special_chars(model->tokenizer->decode(nodes_batch[batch_idx]->ids)).c_str());
            }
        }
        _prefill_progress_finish();

        if (logits == nullptr) {
            if (nodes_batch[batch_idx]->logits.size() == num_vocab) {
                logits = nodes_batch[batch_idx]->logits.data();
            } else {
                LOGE("no logits found, neither from saved state nor from new tokens to prefill\n");
                ret = eval_logits(model_id, text_ids_batch[batch_idx].back(), logits);
                if (ret) {
                    model->is_generating = false;
                    LOGE("failed to eval logits\n");
                    return ret;
                }
            }
        }

        is_pseudo_thinking_batch[batch_idx] = enable_reasoning && model->response_buffer_batch[batch_idx].find("</think>") != std::string::npos;
        model->sampler->apply_penalties(logits, num_vocab);

        logits[61] = -1e9f;
        logits[261] = -1e9f;

        decoded_idx[batch_idx] = model->sampler->sample(logits, num_vocab);

        model->backend->get_state(state_batch[batch_idx]);
    }

    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        model->backend->set_state_on_batch_slot(batch_idx, state_batch[batch_idx]);
    }

    // dynamic batch size
    int current_batch_size = batch_size;
    std::vector<int> active_batch_indices;
    std::vector<int> original_to_active_mapping(batch_size, -1);

    for (int j = 0; j < batch_size; j++) {
        active_batch_indices.push_back(j);
        original_to_active_mapping[j] = j;
    }

    for (int i = 0; i < max_length; i++) {
        if (i != 0) {
            for (int j = 0; j < current_batch_size; j++) {
                int original_j = active_batch_indices[j];
                model->sampler->apply_penalties(logits + j * num_vocab, num_vocab, occurences_batch[original_j],
                    model->sampler->get_token_banned(), model->sampler->get_presence_penalty(),
                    model->sampler->get_frequency_penalty(), model->sampler->get_penalty_decay());

                if (is_pseudo_thinking_batch[original_j] && i == 1 && decoded_idx[j] == 11) {
                    logits[j * num_vocab + 61] = -1e9f;
                }
            }

            decoded_idx = model->sampler->sample_batch(logits, model->backend->get_num_vocab(), model->backend->get_num_vocab(), current_batch_size);
        }

        for (int j = 0; j < current_batch_size; j++) {
            int original_j = active_batch_indices[j];
            if (decoded_idx[j] == 0) {
                model->response_buffer_eos_found_batch[original_j] = true;
                std::any state_end;
                model->backend->get_state_on_batch_slot(j, state_end);
                state_batch[original_j] = std::move(state_end);
            }

            if (!model->response_buffer_eos_found_batch[original_j]) {
                std::string decoded = model->tokenizer->decode(decoded_idx[j]);
                std::string tmp = model->response_buffer_batch[original_j] + decoded;
                for (auto &stop_code : model->stop_codes) {
                    if (enable_reasoning && !thinking_end_tag_found_batch[original_j] && stop_code == "\n\n") {
                        continue;
                    }
                    if (tmp.size() >= stop_code.size() &&
                        tmp.compare(tmp.size() - stop_code.size(), stop_code.size(), stop_code) == 0) {
                        LOGD("stop code found for batch %d: %s\n", original_j, stop_code.c_str());
                        model->response_buffer_eos_found_batch[original_j] = true;
                        std::any state_end;
                        model->backend->get_state_on_batch_slot(j, state_end);
                        state_batch[original_j] = std::move(state_end);
                        break;
                    }
                }

                if (enable_reasoning && !thinking_end_tag_found_batch[original_j]) {
                    if (tmp.find("</think>") != std::string::npos) {
                        thinking_end_tag_found_batch[original_j] = true;
                    }
                } else if (state_for_rewinding_batch[original_j].has_value()) {
                    state_for_rewinding_batch[original_j].reset();
                }

                if (std::any_of(rewind_token_list, rewind_token_list + sizeof(rewind_token_list) / sizeof(rewind_token_list[0]), [decoded_idx, j](int token) { return decoded_idx[j] == token; })) {
                    model->backend->get_state_on_batch_slot(j, state_for_rewinding_batch[original_j]);
                }
            }
        }

        std::vector<int> new_active_batch_indices;
        for (int j = 0; j < batch_size; j++) {
            if (!model->response_buffer_eos_found_batch[j]) {
                new_active_batch_indices.push_back(j);
            }
        }

        int new_active_count = new_active_batch_indices.size();
        if (new_active_count > 0 && new_active_count < current_batch_size) {
            bool new_size_supported = false;
            for (auto size : model->backend->supported_batch_sizes) {
                if (new_active_count == size) {
                    new_size_supported = true;
                    break;
                }
            }

            if (new_size_supported) {
                LOGI("Switching from batch size %d to %d, active batches: ", current_batch_size, new_active_count);
                for (auto idx : new_active_batch_indices) {
                    LOGI("%d ", idx);
                }

                // get state for new active batches
                std::vector<std::any> temp_states(new_active_count);
                for (int k = 0; k < new_active_count; k++) {
                    int original_batch_idx = new_active_batch_indices[k];
                    int current_slot_idx = original_to_active_mapping[original_batch_idx];
                    if (current_slot_idx >= 0) {
                        model->backend->get_state_on_batch_slot(current_slot_idx, temp_states[k]);
                    }
                }

                // rearrange logits for new active batches
                std::vector<float> temp_logits(new_active_count * num_vocab);
                for (int k = 0; k < new_active_count; k++) {
                    int original_batch_idx = new_active_batch_indices[k];
                    int current_slot_idx = original_to_active_mapping[original_batch_idx];
                    if (current_slot_idx >= 0 && current_slot_idx < current_batch_size) {
                        memcpy(temp_logits.data() + k * num_vocab, 
                            logits + current_slot_idx * num_vocab, 
                            num_vocab * sizeof(float));
                    }
                }
                // copy rearranged logits back
                memcpy(logits, temp_logits.data(), new_active_count * num_vocab * sizeof(float));

                // rearrange decoded_idx for new active batches
                std::vector<int> temp_decoded_idx(new_active_count);
                for (int k = 0; k < new_active_count; k++) {
                    int original_batch_idx = new_active_batch_indices[k];
                    int current_slot_idx = original_to_active_mapping[original_batch_idx];
                    if (current_slot_idx >= 0 && current_slot_idx < current_batch_size) {
                        temp_decoded_idx[k] = decoded_idx[current_slot_idx];
                        LOGD("Rearranging decoded_idx[%d]: original_batch=%d, current_slot=%d, token=%d\n", 
                             k, original_batch_idx, current_slot_idx, decoded_idx[current_slot_idx]);
                    } else {
                        LOGE("Invalid mapping for batch %d: current_slot=%d (should be in [0,%d))\n", 
                             original_batch_idx, current_slot_idx, current_batch_size);
                        temp_decoded_idx[k] = 0; // fallback to EOS to avoid undefined behavior
                    }
                }
                // copy rearranged decoded_idx back
                for (int k = 0; k < new_active_count; k++) {
                    decoded_idx[k] = temp_decoded_idx[k];
                }

                // rearrange active state to corresponding slots
                for (int k = 0; k < new_active_count; k++) {
                    model->backend->set_state_on_batch_slot(k, temp_states[k]);
                    model->backend->free_state(temp_states[k]);
                }

                // update active batch indices and mapping
                active_batch_indices = new_active_batch_indices;
                current_batch_size = new_active_count;

                original_to_active_mapping.assign(batch_size, -1);
                for (int k = 0; k < new_active_count; k++) {
                    original_to_active_mapping[active_batch_indices[k]] = k;
                }
            }
        }

        auto all_eos_found = std::all_of(model->response_buffer_eos_found_batch.begin(), model->response_buffer_eos_found_batch.end(), [](bool eos_found) { return eos_found; });
        if (all_eos_found || model->stop_signal) {
            LOGD("stopping generation, eos_found: %d, stop_signal: %d\n", all_eos_found, model->stop_signal);
            break;
        }

        std::vector<int> active_decoded_idx(decoded_idx.begin(), decoded_idx.begin() + current_batch_size);
        ret = eval_logits_batch_decode(model_id, active_decoded_idx, logits);
        if (ret) {
            model->is_generating = false;
            LOGE("failed to eval logits\n");
            return ret;
        }

        for (int j = 0; j < current_batch_size; j++) {
            int original_j = active_batch_indices[j];
            if (model->response_buffer_eos_found_batch[original_j]) {
                continue;
            }
            response_ids_raw_batch[original_j].emplace_back(decoded_idx[j]);
            model->response_buffer_batch[original_j] += model->tokenizer->decode(decoded_idx[j]);
            model->response_buffer_ids_batch[original_j].emplace_back(decoded_idx[j]);
            if (i == 0 && model->response_buffer_batch[original_j][0] == ' ') {
                model->response_buffer_batch[original_j] = model->response_buffer_batch[original_j].substr(1);
            }

            occurences_batch[original_j][decoded_idx[j]]++;
        }

        // TODO: callback_batch
        // if (callback) {
        //     callback(model->response_buffer.c_str(), decoded_idx, decoded.c_str());
        // }
    }

    // save state for not finished batches
    for (int j = 0; j < batch_size; j++) {
        if (!model->response_buffer_eos_found_batch[j] && !state_batch[j].has_value()) {
            int active_slot = original_to_active_mapping[j];
            if (active_slot >= 0) {
                model->backend->get_state_on_batch_slot(active_slot, state_batch[j]);
            }
        }
    }

    if (response_ids_raw_batch[0].size() > 0) {
        for (int j = 0; j < batch_size; j++) {
            if (state_for_rewinding_batch[j].has_value()) {
                response_ids_raw_batch[j].pop_back();
                state_batch[j] = std::move(state_for_rewinding_batch[j]);
            }
        }
        int ret = model->backend->register_batch_state_checkpoint(nodes_batch, state_batch, response_ids_raw_batch, logits);
        if (ret) {
            model->is_generating = false;
            LOGE("failed to register batch state checkpoint\n");
            return ret;
        }
    }

    model->is_generating = false;
    model->stop_signal = false;

    model->backend->cleanup_state_tree();

    return RWKV_SUCCESS;
}

int runtime::set_prompt(int model_id, std::string prompt) {
    if (_models.find(model_id) == _models.end()) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);
    if (model->backend == nullptr || model->tokenizer == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    LOGD("Setting and processing prompt for model %d: \"%s\"\n", model_id, prompt.c_str());
    std::vector<int> ids = model->tokenizer->encode(prompt);
    if (ids.empty()) {
        LOGD("Got empty prompt\n");
        return RWKV_SUCCESS;
    }
    std::vector<int> new_ids_to_prefill;
    auto node = model->backend->match_and_load_state(ids, new_ids_to_prefill);
    if (new_ids_to_prefill.empty()) {
        return RWKV_SUCCESS;
    }

    model->prompt = prompt;
    model->backend->set_state(node->state);

    float *logits = nullptr;
    int ret = eval_logits(model_id, ids, logits);
    if (ret) {
        return ret;
    }
    model->backend->register_state_checkpoint(node, ids, logits);
    return RWKV_SUCCESS;
}

std::string runtime::get_prompt(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return "";
    }
    auto &model = _models.at(model_id);
    return model->prompt;
}

#ifdef ENABLE_WHISPER
int runtime::set_audio_prompt(int model_id, std::string path) {
    if (_models.find(model_id) == _models.end()) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);
    if (model->backend == nullptr || model->tokenizer == nullptr || model->multimodal_encoder == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    std::string prompt = "<audio src=\"" + path + "\">";
    std::vector<int> ids = model->tokenizer->encode(prompt);

    std::vector<int> new_ids_to_prefill;
    auto node = model->backend->match_and_load_state(ids, new_ids_to_prefill);
    if (new_ids_to_prefill.empty()) {
        return RWKV_SUCCESS;
    }

    model->prompt = prompt;
    model->backend->set_state(node->state);

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<float> embeddings;
    int n_tokens;
    if (!model->multimodal_encoder->Encode(path, embeddings, n_tokens, model->backend->embedding_input_force_no_ln0())) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto end = std::chrono::high_resolution_clock::now();
    LOGI("whisper duration: %lld ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    float *logits = nullptr;

    int ret = eval_logits_with_embeddings(model_id, embeddings.data(), n_tokens, logits);
    if (ret) {
        LOGE("%s failed to eval logits with embeddings\n", __func__);
        return ret;
    }
    model->backend->register_state_checkpoint(node, ids, logits);
    return RWKV_SUCCESS;
}
#endif

#ifdef ENABLE_TTS
namespace {
int generate_tts_output(
    runtime* rt,
    int model_id,
    NucleusSampler* sampler,
    execution_provider* backend,
    float*& logits,
    std::vector<int>& output_tokens
) {
    static const int tts_max_length = 3000;
    static const int tts_top_k = 50;
    static const float tts_top_p = 0.95;
    static const float tts_temperature = 1.0;
    static const int tts_eos_token = 8192;
    static const int tts_tag_token_offset = 8193;

    for (int i = 0; i < tts_max_length; i++) {
        int idx = sampler->sample(logits, tts_tag_token_offset, tts_temperature, tts_top_k, tts_top_p);
        if (idx == tts_eos_token) {
            break;
        }

        output_tokens.push_back(idx);
        int ret = rt->eval_logits(model_id, idx, logits);
        if (ret || !logits) {
            LOGE("[TTS] Error evaluating logits");
            return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
        }
    }
    return RWKV_SUCCESS;
}

void tts_detokenize_thread_main(
    sparktts* sparktts_instance,
    runtime* rt,
    const std::vector<int>& global_tokens,
    const std::vector<int>& output_tokens,
    const bool& generation_finished,
    double& ttfa,
    const std::chrono::high_resolution_clock::time_point& total_start
) {
    int actual_chunk_size = sparktts_instance->initial_chunk_size;
    std::vector<int> semantic_tokens_buf;
    sparktts_instance->resize_detokenizer_model(actual_chunk_size);
    int semantic_token_pos = 0;
    while (!generation_finished) {
        if (output_tokens.size() - semantic_token_pos >= actual_chunk_size) {
            std::vector<int> current_chunk_tokens(output_tokens.begin() + semantic_token_pos, output_tokens.begin() + semantic_token_pos + actual_chunk_size);
            semantic_token_pos += actual_chunk_size;
            int buffered_size = semantic_tokens_buf.size();
            std::vector<int> current_semantic_tokens = semantic_tokens_buf;
            current_semantic_tokens.insert(current_semantic_tokens.end(), current_chunk_tokens.begin(), current_chunk_tokens.end());
            semantic_tokens_buf = std::vector<int>(current_semantic_tokens.begin() + (current_semantic_tokens.size() - sparktts_instance->overlap_size), current_semantic_tokens.end());
            auto new_samples = sparktts_instance->detokenize_audio(global_tokens, current_semantic_tokens);
            if (rt->tts_get_streaming_buffer().empty()) {
                ttfa = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - total_start).count();
                actual_chunk_size = sparktts_instance->chunk_size;
                sparktts_instance->resize_detokenizer_model(actual_chunk_size);
            }
            // tts_output_samples_buffer.insert(tts_output_samples_buffer.end(), new_samples.begin() + (16000 * buffered_size / 50), new_samples.end());
            rt->tts_append_samples_to_buffer(new_samples.begin() + (16000 * buffered_size / 50), new_samples.end());
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    if (output_tokens.size() - semantic_token_pos > 0) {
        std::vector<int> current_chunk_tokens(output_tokens.begin() + semantic_token_pos, output_tokens.end());
        int buffered_size = semantic_tokens_buf.size();
        std::vector<int> current_semantic_tokens = semantic_tokens_buf;
        current_semantic_tokens.insert(current_semantic_tokens.end(), current_chunk_tokens.begin(), current_chunk_tokens.end());
        auto new_samples = sparktts_instance->detokenize_audio(global_tokens, current_semantic_tokens);
        // tts_output_samples_buffer.insert(tts_output_samples_buffer.end(), new_samples.begin() + (16000 * buffered_size / 50), new_samples.end());
        rt->tts_append_samples_to_buffer(new_samples.begin() + (16000 * buffered_size / 50), new_samples.end());
    }
}
} // anonymous namespace

int runtime::sparktts_load_models(
    std::string wav2vec2_path,
    std::string bicodec_tokenizer_path,
    std::string bicodec_detokenizer_path
) {
    _sparktts = std::make_unique<sparktts>();
    if (!_sparktts->load_models(wav2vec2_path, bicodec_tokenizer_path, bicodec_detokenizer_path)) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    return RWKV_SUCCESS;
}

int runtime::sparktts_release_models() {
    _sparktts = nullptr;
    return RWKV_SUCCESS;
}

int runtime::run_spark_tts_zeroshot(int model_id, std::string tts_text, std::string prompt_audio_text, std::string prompt_audio_path, std::string output_wav_path) {
    if (_sparktts == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);

    static const int tts_tag_token_offset = 8193;
    static const int global_token_offset = 8196;

    wav_file wav;
    wav.load(prompt_audio_path);
    wav.resample(16000);

    auto total_start = std::chrono::high_resolution_clock::now();
    std::vector<int> global_tokens;
    std::vector<int> semantic_tokens;
    _sparktts->tokenize_audio(wav.samples, global_tokens, semantic_tokens);
    if (prompt_audio_text.empty()) {
        semantic_tokens.clear();
    }

    std::string full_text = prompt_audio_text + tts_text;
    auto text_tokens = tokenizer_encode(model_id, full_text);
    if (text_tokens.empty()) {
        LOGE("[TTS] Text tokenizer encode failed");
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    std::vector<int> input_tokens = {tts_tag_token_offset + 2}; // tag_2
    for (int i = 0; i < text_tokens.size(); i++) {
        input_tokens.push_back(text_tokens[i]);
    }
    input_tokens.push_back(tts_tag_token_offset + 0); // tag_0
    for (int i = 0; i < global_tokens.size(); i++) {
        input_tokens.push_back(global_tokens[i] + global_token_offset);
    }
    input_tokens.push_back(tts_tag_token_offset + 1); // tag_1
    for (int i = 0; i < semantic_tokens.size(); i++) {
        input_tokens.push_back(semantic_tokens[i]);
    }

    _global_tokens_output.clear();
    for (auto token : global_tokens) {
        _global_tokens_output.push_back(token + global_token_offset);
    }

    std::vector<int> output_tokens;

    auto start = std::chrono::high_resolution_clock::now();

    clear_state(model_id);
    float *logits = nullptr;
    int ret = eval_logits(model_id, input_tokens, logits);
    if (ret || !logits) {
        LOGE("[TTS] Error evaluating logits");
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    ret = generate_tts_output(this, model_id, model->sampler.get(), model->backend.get(), logits, output_tokens);
    if (ret) return ret;

    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    LOGI("[TTS] LLM inference time: %lf ms", duration);
    LOGI("[TTS] LLM output tokens: %d", output_tokens.size());
    LOGI("[TTS] LLM prefill speed: %f tokens/s", get_avg_prefill_speed(model_id));
    LOGI("[TTS] LLM decode speed: %f tokens/s", get_avg_decode_speed(model_id));

    std::vector<float> output_samples = _sparktts->detokenize_audio(global_tokens, output_tokens);
    save_samples_to_wav(output_samples, output_wav_path, 16000);

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
    LOGI("[TTS] Total time: %lf ms", total_duration);
    LOGI("[TTS] Output audio length: %lf s", output_samples.size() / 16000.0);
    LOGI("[TTS] RTF: %lf", total_duration / 1e3f * 16000.0 / output_samples.size());

    set_is_generating(model_id, false);
    return RWKV_SUCCESS;
}

int runtime::run_spark_tts_with_properties(int model_id, std::string tts_text, std::string output_wav_path,
    std::string age, std::string gender, std::string emotion, std::string pitch, std::string speed) {
    if (_sparktts == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);

    static const int tts_tag_token_offset = 8193;
    static const int global_token_offset = 8196;

    std::vector<int> properties_tokens = convert_standard_properties_to_tokens(age, gender, emotion, pitch, speed);

    auto total_start = std::chrono::high_resolution_clock::now();
    std::vector<int> global_tokens;
    std::vector<int> semantic_tokens;

    auto text_tokens = tokenizer_encode(model_id, tts_text);
    if (text_tokens.empty()) {
        LOGE("[TTS] Text tokenizer encode failed");
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    std::vector<int> input_tokens = properties_tokens;
    input_tokens.push_back(tts_tag_token_offset + 2); // tag_2
    for (int i = 0; i < text_tokens.size(); i++) {
        input_tokens.push_back(text_tokens[i]);
    }
    input_tokens.push_back(tts_tag_token_offset + 0); // tag_0

    clear_state(model_id);
    float *logits = nullptr;
    int ret = eval_logits(model_id, input_tokens, logits);
    if (ret || !logits) {
        LOGE("[TTS] Error evaluating logits");
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    for (int i = 0; i < 32; i++) { // generate 32 global_tokens
        int idx = model->sampler->sample(logits, 4096, 1.0, 20, 0.95);

        global_tokens.push_back(idx + global_token_offset);
        ret = eval_logits(model_id, idx + global_token_offset, logits);
        if (ret || !logits) {
            LOGE("[TTS] Error evaluating logits");
            return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
        }
    }

    _global_tokens_output = global_tokens;

    ret = eval_logits(model_id, tts_tag_token_offset + 1, logits);
    if (ret || !logits) {
        LOGE("[TTS] Error evaluating logits");
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    std::vector<int> output_tokens;

    ret = generate_tts_output(this, model_id, model->sampler.get(), model->backend.get(), logits, output_tokens);
    if (ret) return ret;

    LOGI("[TTS] LLM output tokens: %d", output_tokens.size());
    LOGI("[TTS] LLM prefill speed: %f tokens/s", get_avg_prefill_speed(model_id));
    LOGI("[TTS] LLM decode speed: %f tokens/s", get_avg_decode_speed(model_id));

    std::vector<float> output_samples = _sparktts->detokenize_audio(global_tokens, output_tokens);
    save_samples_to_wav(output_samples, output_wav_path, 16000);

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
    LOGI("[TTS] Total time: %lf ms", total_duration);
    LOGI("[TTS] Output audio length: %lf s", output_samples.size() / 16000.0);
    LOGI("[TTS] RTF: %lf", total_duration / 1e3f * 16000.0 / output_samples.size());

    set_is_generating(model_id, false);
    return RWKV_SUCCESS;
}

int runtime::run_spark_tts_with_global_tokens(int model_id, std::string tts_text, std::string output_wav_path, std::vector<int> global_tokens) {
    if (_sparktts == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);

    static const int tts_tag_token_offset = 8193;

    auto total_start = std::chrono::high_resolution_clock::now();

    auto text_tokens = tokenizer_encode(model_id, tts_text);
    if (text_tokens.empty()) {
        LOGE("[TTS] Text tokenizer encode failed");
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    std::vector<int> input_tokens;
    input_tokens.push_back(tts_tag_token_offset + 2); // tag_2
    for (int i = 0; i < text_tokens.size(); i++) {
        input_tokens.push_back(text_tokens[i]);
    }
    input_tokens.push_back(tts_tag_token_offset + 0); // tag_0
    for (int i = 0; i < global_tokens.size(); i++) {
        input_tokens.push_back(global_tokens[i]); // global_tokens are already offset by global_token_offset
    }
    input_tokens.push_back(tts_tag_token_offset + 1); // tag_1

    clear_state(model_id);
    float *logits = nullptr;
    int ret = eval_logits(model_id, input_tokens, logits);
    if (ret || !logits) {
        LOGE("[TTS] Error evaluating logits");
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    std::vector<int> output_tokens;

    ret = generate_tts_output(this, model_id, model->sampler.get(), model->backend.get(), logits, output_tokens);
    if (ret) return ret;

    LOGI("[TTS] LLM output tokens: %d", output_tokens.size());
    LOGI("[TTS] LLM prefill speed: %f tokens/s", get_avg_prefill_speed(model_id));
    LOGI("[TTS] LLM decode speed: %f tokens/s", get_avg_decode_speed(model_id));

    std::vector<float> output_samples = _sparktts->detokenize_audio(global_tokens, output_tokens);
    save_samples_to_wav(output_samples, output_wav_path, 16000);

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
    LOGI("[TTS] Total time: %lf ms", total_duration);
    LOGI("[TTS] Output audio length: %lf s", output_samples.size() / 16000.0);
    LOGI("[TTS] RTF: %lf", total_duration / 1e3f * 16000.0 / output_samples.size());

    set_is_generating(model_id, false);
    return RWKV_SUCCESS;
}

int runtime::run_spark_tts_zeroshot_streaming(int model_id, std::string tts_text, std::string prompt_audio_text, std::string prompt_audio_path, std::string output_wav_path) {
    if (_sparktts == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

#if !defined(_WIN32)
    auto texts = tts_frontend_utils::process_text(tts_text,
        [this, model_id](const std::string& text) -> std::vector<int> {
            return tokenizer_encode(model_id, text);
        },
        _tn_list
    );
#else
    auto texts = tts_frontend_utils::process_text(tts_text,
        [this, model_id](const std::string& text) -> std::vector<int> {
            return tokenizer_encode(model_id, text);
        }
    );
#endif

    tts_clear_streaming_buffer();
    auto total_start = std::chrono::high_resolution_clock::now();
    std::vector<int> global_tokens;
    std::vector<int> semantic_tokens;

    bool read_from_cache = _sparktts->get_global_and_semantic_tokens(prompt_audio_path, _cache_dir, global_tokens, semantic_tokens);
    if (semantic_tokens.empty() || global_tokens.empty()) {
        LOGE("[TTS] Failed to get global and/or semantic tokens");
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    static const int global_token_offset = 8196;
    _global_tokens_output.clear();
    for (auto token : global_tokens) {
        _global_tokens_output.push_back(token + global_token_offset);
    }

    if (prompt_audio_text.empty()) {
        semantic_tokens.clear();
    }

    // variables between threads
    bool generation_finished = false;
    std::vector<int> output_tokens;

    auto llm_inference_thread = std::thread([&]() {
        auto &model = _models.at(model_id);
        static const int tts_tag_token_offset = 8193;
        static const int global_token_offset = 8196;

        static const int tts_max_length = 3000;
        static const int tts_top_k = 50;
        static const float tts_top_p = 0.95;
        static const float tts_temperature = 1.0;
        static const int tts_eos_token = 8192;
        for (auto &text : texts) {
            std::string full_text = prompt_audio_text + text;
            LOGI("[TTS] LLM input text: %s", full_text.c_str());
            auto text_tokens = tokenizer_encode(model_id, full_text);
            if (text_tokens.empty()) {
                LOGE("[TTS] Text tokenizer encode failed");
                generation_finished = true;
                return;
            }

            std::vector<int> input_tokens = {tts_tag_token_offset + 2}; // tag_2
            for (int i = 0; i < text_tokens.size(); i++) {
                input_tokens.push_back(text_tokens[i]);
            }
            input_tokens.push_back(tts_tag_token_offset + 0); // tag_0
            for (int i = 0; i < global_tokens.size(); i++) {
                input_tokens.push_back(global_tokens[i] + global_token_offset);
            }
            input_tokens.push_back(tts_tag_token_offset + 1); // tag_1
            for (int i = 0; i < semantic_tokens.size(); i++) {
                input_tokens.push_back(semantic_tokens[i]);
            }

            clear_state(model_id);
            float *logits = nullptr;
            int ret = eval_logits(model_id, input_tokens, logits);
            if (ret || !logits) {
                LOGE("[TTS] Error evaluating logits");
                generation_finished = true;
                return;
            }
            logits[tts_eos_token] = -1e9;

            for (int i = 0; i < tts_max_length; i++) {
                int idx = model->sampler->sample(logits, tts_tag_token_offset, tts_temperature, tts_top_k, tts_top_p);
                if (idx == tts_eos_token) {
                    LOGI("[TTS] EOS token found");
                    break;
                }

                output_tokens.push_back(idx);
                ret = eval_logits(model_id, idx, logits);
                if (ret || !logits) {
                    LOGE("[TTS] Error evaluating logits");
                    generation_finished = true;
                    return;
                }
            }
        }
        generation_finished = true;
    });

    double ttfa = 0.0;
    std::thread detokenize_thread([&]() {
        tts_detokenize_thread_main(_sparktts.get(), this, global_tokens, output_tokens, generation_finished, ttfa, total_start);
    });

    llm_inference_thread.join();
    detokenize_thread.join();

    LOGI("[TTS] LLM output tokens: %d", output_tokens.size());
    LOGI("[TTS] LLM prefill speed: %f tokens/s", get_avg_prefill_speed(model_id));
    LOGI("[TTS] LLM decode speed: %f tokens/s", get_avg_decode_speed(model_id));
    if (!_tts_output_samples_buffer.empty()) {
        save_samples_to_wav(_tts_output_samples_buffer, output_wav_path, 16000);
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
    LOGI("[TTS] Total time (%s): %lf ms", read_from_cache ? "prompt audio tokens cache hit" : "prompt audio tokens cache miss", total_duration);
    LOGI("[TTS] Output audio length: %lf s", _tts_output_samples_buffer.size() / 16000.0);
    LOGI("[TTS] RTF (%s): %lf", read_from_cache ? "prompt audio tokens cache hit" : "prompt audio tokens cache miss", total_duration / 1e3f * 16000.0 / _tts_output_samples_buffer.size());
    LOGI("[TTS] TTFA (%s): %lf ms", read_from_cache ? "prompt audio tokens cache hit" : "prompt audio tokens cache miss", ttfa);

    set_is_generating(model_id, false);
    return RWKV_SUCCESS;
}

int runtime::run_spark_tts_with_properties_streaming(int model_id, std::string tts_text, std::string output_wav_path,
    std::string age, std::string gender, std::string emotion, std::string pitch, std::string speed) {
    if (_sparktts == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

#if !defined(_WIN32)
    auto texts = tts_frontend_utils::process_text(tts_text,
        [this, model_id](const std::string& text) -> std::vector<int> {
            return tokenizer_encode(model_id, text);
        },
        _tn_list
    );
#else
    auto texts = tts_frontend_utils::process_text(tts_text,
        [this, model_id](const std::string& text) -> std::vector<int> {
            return tokenizer_encode(model_id, text);
        }
    );
#endif

    tts_clear_streaming_buffer();
    auto total_start = std::chrono::high_resolution_clock::now();
    std::vector<int> global_tokens;
    std::vector<int> properties_tokens = convert_standard_properties_to_tokens(age, gender, emotion, pitch, speed);

    // variables between threads
    bool generation_finished = false;
    std::vector<int> output_tokens;

    auto llm_inference_thread = std::thread([&]() {
        auto &model = _models.at(model_id);
        static const int tts_tag_token_offset = 8193;
        static const int global_token_offset = 8196;

        static const int tts_max_length = 3000;
        static const int tts_top_k = 50;
        static const float tts_top_p = 0.95;
        static const float tts_temperature = 1.0;
        static const int tts_eos_token = 8192;
        for (auto &text : texts) {
            LOGI("[TTS] LLM input text: %s", text.c_str());
            auto text_tokens = tokenizer_encode(model_id, text);
            if (text_tokens.empty()) {
                LOGE("[TTS] Text tokenizer encode failed");
                generation_finished = true;
                return;
            }

            clear_state(model_id);
            float *logits = nullptr;

            if (global_tokens.empty()) {
                // generate global tokens
                std::vector<int> input_tokens = properties_tokens;
                input_tokens.push_back(tts_tag_token_offset + 2); // tag_2
                for (int i = 0; i < text_tokens.size(); i++) {
                    input_tokens.push_back(text_tokens[i]);
                }
                input_tokens.push_back(tts_tag_token_offset + 0); // tag_0

                int ret = eval_logits(model_id, input_tokens, logits);
                if (ret || !logits) {
                    LOGE("[TTS] Error evaluating logits");
                    generation_finished = true;
                    return;
                }

                for (int i = 0; i < 32; i++) { // generate 32 global_tokens
                    int idx = model->sampler->sample(logits, 4096, 1.0, 20, 0.95);

                    global_tokens.push_back(idx + global_token_offset);
                    ret = eval_logits(model_id, idx + global_token_offset, logits);
                    if (ret || !logits) {
                        LOGE("[TTS] Error evaluating logits");
                        generation_finished = true;
                        return;
                    }
                }

                _global_tokens_output = global_tokens;

                ret = eval_logits(model_id, tts_tag_token_offset + 1, logits);
                if (ret || !logits) {
                    LOGE("[TTS] Error evaluating logits");
                    generation_finished = true;
                    return;
                }
            } else {
                std::vector<int> input_tokens = {tts_tag_token_offset + 2}; // tag_2
                for (int i = 0; i < text_tokens.size(); i++) {
                    input_tokens.push_back(text_tokens[i]);
                }
                input_tokens.push_back(tts_tag_token_offset + 0); // tag_0
                for (int i = 0; i < global_tokens.size(); i++) {
                    input_tokens.push_back(global_tokens[i] + global_token_offset);
                }
                input_tokens.push_back(tts_tag_token_offset + 1); // tag_1

                int ret = eval_logits(model_id, input_tokens, logits);
                if (ret || !logits) {
                    LOGE("[TTS] Error evaluating logits");
                    generation_finished = true;
                    return;
                }
            }

            logits[tts_eos_token] = -1e9;

            for (int i = 0; i < tts_max_length; i++) {
                int idx = model->sampler->sample(logits, tts_tag_token_offset, tts_temperature, tts_top_k, tts_top_p);
                if (idx == tts_eos_token) {
                    LOGI("[TTS] EOS token found");
                    break;
                }

                output_tokens.push_back(idx);
                int ret = eval_logits(model_id, idx, logits);
                if (ret || !logits) {
                    LOGE("[TTS] Error evaluating logits");
                    generation_finished = true;
                    return;
                }
            }
        }
        generation_finished = true;
    });

    double ttfa = 0.0;
    std::thread detokenize_thread([&]() {
        tts_detokenize_thread_main(_sparktts.get(), this, global_tokens, output_tokens, generation_finished, ttfa, total_start);
    });

    llm_inference_thread.join();
    detokenize_thread.join();

    LOGI("[TTS] %s: LLM output tokens: %d", __func__, output_tokens.size());
    LOGI("[TTS] %s: LLM prefill speed: %f tokens/s", __func__, get_avg_prefill_speed(model_id));
    LOGI("[TTS] %s: LLM decode speed: %f tokens/s", __func__, get_avg_decode_speed(model_id));
    if (!_tts_output_samples_buffer.empty()) {
        save_samples_to_wav(_tts_output_samples_buffer, output_wav_path, 16000);
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
    LOGI("[TTS] %s: Total time: %lf ms", __func__, total_duration);
    LOGI("[TTS] %s: Output audio length: %lf s", __func__, _tts_output_samples_buffer.size() / 16000.0);
    LOGI("[TTS] %s: RTF: %lf", __func__, total_duration / 1e3f * 16000.0 / _tts_output_samples_buffer.size());
    LOGI("[TTS] %s: TTFA: %lf ms", __func__, ttfa);

    set_is_generating(model_id, false);
    return RWKV_SUCCESS;
}


int runtime::run_spark_tts_with_global_tokens_streaming(int model_id, std::string tts_text, std::string output_wav_path, std::vector<int> global_tokens) {
    if (_sparktts == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

#if !defined(_WIN32)
    auto texts = tts_frontend_utils::process_text(tts_text,
        [this, model_id](const std::string& text) -> std::vector<int> {
            return tokenizer_encode(model_id, text);
        },
        _tn_list
    );
#else
    auto texts = tts_frontend_utils::process_text(tts_text,
        [this, model_id](const std::string& text) -> std::vector<int> {
            return tokenizer_encode(model_id, text);
        }
    );
#endif

    tts_clear_streaming_buffer();
    auto total_start = std::chrono::high_resolution_clock::now();

    // variables between threads
    bool generation_finished = false;
    std::vector<int> output_tokens;

    auto llm_inference_thread = std::thread([&]() {
        auto &model = _models.at(model_id);
        static const int tts_tag_token_offset = 8193;
        static const int global_token_offset = 8196;

        static const int tts_max_length = 3000;
        static const int tts_top_k = 50;
        static const float tts_top_p = 0.95;
        static const float tts_temperature = 1.0;
        static const int tts_eos_token = 8192;
        for (auto &text : texts) {
            LOGI("[TTS] LLM input text: %s", text.c_str());
            auto text_tokens = tokenizer_encode(model_id, text);
            if (text_tokens.empty()) {
                LOGE("[TTS] Text tokenizer encode failed");
                generation_finished = true;
                return;
            }

            clear_state(model_id);
            float *logits = nullptr;

            std::vector<int> input_tokens = {tts_tag_token_offset + 2}; // tag_2
            for (int i = 0; i < text_tokens.size(); i++) {
                input_tokens.push_back(text_tokens[i]);
            }
            input_tokens.push_back(tts_tag_token_offset + 0); // tag_0
            for (int i = 0; i < global_tokens.size(); i++) {
                input_tokens.push_back(global_tokens[i]); // global_tokens are already offset by global_token_offset
            }
            input_tokens.push_back(tts_tag_token_offset + 1); // tag_1

            int ret = eval_logits(model_id, input_tokens, logits);
            if (ret || !logits) {
                LOGE("[TTS] Error evaluating logits");
                generation_finished = true;
                return;
            }

            logits[tts_eos_token] = -1e9;

            for (int i = 0; i < tts_max_length; i++) {
                int idx = model->sampler->sample(logits, tts_tag_token_offset, tts_temperature, tts_top_k, tts_top_p);
                if (idx == tts_eos_token) {
                    LOGI("[TTS] EOS token found");
                    break;
                }

                output_tokens.push_back(idx);
                int ret = eval_logits(model_id, idx, logits);
                if (ret || !logits) {
                    LOGE("[TTS] Error evaluating logits");
                    generation_finished = true;
                    return;
                }
            }
        }
        generation_finished = true;
    });

    double ttfa = 0.0;
    std::thread detokenize_thread([&]() {
        tts_detokenize_thread_main(_sparktts.get(), this, global_tokens, output_tokens, generation_finished, ttfa, total_start);
    });

    llm_inference_thread.join();
    detokenize_thread.join();

    LOGI("[TTS] %s: LLM output tokens: %d", __func__, output_tokens.size());
    LOGI("[TTS] %s: LLM prefill speed: %f tokens/s", __func__, get_avg_prefill_speed(model_id));
    LOGI("[TTS] %s: LLM decode speed: %f tokens/s", __func__, get_avg_decode_speed(model_id));
    if (!_tts_output_samples_buffer.empty()) {
        save_samples_to_wav(_tts_output_samples_buffer, output_wav_path, 16000);
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
    LOGI("[TTS] %s: Total time: %lf ms", __func__, total_duration);
    LOGI("[TTS] %s: Output audio length: %lf s", __func__, _tts_output_samples_buffer.size() / 16000.0);
    LOGI("[TTS] %s: RTF: %lf", __func__, total_duration / 1e3f * 16000.0 / _tts_output_samples_buffer.size());
    LOGI("[TTS] %s: TTFA: %lf ms", __func__, ttfa);

    set_is_generating(model_id, false);
    return RWKV_SUCCESS;
}
#endif

int runtime::clear_state(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);

    if (model->backend != nullptr) {
        model->backend->zero_state();
        int max_supported_batch_size = *std::max_element(model->backend->supported_batch_sizes.begin(), model->backend->supported_batch_sizes.end());
        if (max_supported_batch_size > 1) {
            for (int i = 0; i < max_supported_batch_size; i++) {
                model->backend->zero_state_on_batch_slot(i);
            }
        }
    }
    return RWKV_SUCCESS;
}

int runtime::gen_completion_batch(int model_id, std::vector<std::string> prompts, int max_length, int batch_size, int stop_code, void (*callback_batch)(const int, const char **, const int*, const char **)) {
    if (_models.find(model_id) == _models.end()) {
        LOGE("gen_completion_batch: Model ID %d not found", model_id);
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);
    if (model->backend == nullptr || model->tokenizer == nullptr) {
        LOGE("gen_completion_batch: Backend or tokenizer for model ID %d not found", model_id);
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    bool supported = false;
    for (auto size : model->backend->supported_batch_sizes) {
        if (batch_size == size) {
            supported = true;
            break;
        }
    }
    if (!supported) {
        LOGE("chat_batch: batch size %d is not supported\n", batch_size);
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_UNSUPPORTED;
    }

    if (prompts.size() != batch_size) {
        LOGE("chat_batch: prompts size %d is not equal to batch size %d\n", prompts.size(), batch_size);
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    model->is_generating = true;
    model->stop_signal = false;

    model->response_buffer_batch.resize(batch_size);
    model->response_buffer_ids_batch.resize(batch_size);
    model->response_buffer_eos_found_batch.resize(batch_size);

    std::vector<int> decoded_idx_batch(batch_size);
    std::vector<std::string> decoded_text_batch(batch_size);
    std::vector<std::any> state_batch(batch_size);
    float *logits = nullptr;

    std::vector<std::map<int, float>> occurences_batch(batch_size);
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        model->backend->get_state_on_batch_slot(batch_idx, state_batch[batch_idx]);
    }

    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        std::vector<int> ids = model->tokenizer->encode(prompts[batch_idx]);
        _prefill_progress_start(ids.size());
        model->backend->set_state(state_batch[batch_idx]);

        int ret = eval_logits(model_id, ids, logits);
        if (ret || !logits) {
            LOGE("gen_completion_batch: Error evaluating logits");
            model->is_generating = false;
            return ret;
        }
        _prefill_progress_finish();

        model->backend->get_state(state_batch[batch_idx]);

        model->response_buffer_batch[batch_idx] = prompts[batch_idx];
        model->response_buffer_ids_batch[batch_idx] = ids;

        model->sampler->apply_penalties(logits, model->backend->get_num_vocab(), occurences_batch[batch_idx],
            model->sampler->get_token_banned(), model->sampler->get_presence_penalty(),
            model->sampler->get_frequency_penalty(), model->sampler->get_penalty_decay());
        decoded_idx_batch[batch_idx] = model->sampler->sample(logits, model->backend->get_num_vocab());
    }

    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        model->backend->set_state_on_batch_slot(batch_idx, state_batch[batch_idx]);
    }

    for (int i = 0; i < max_length; i++) {
        if (i != 0) {
            for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
                model->sampler->apply_penalties(logits + batch_idx * model->backend->get_num_vocab(), model->backend->get_num_vocab(), occurences_batch[batch_idx],
                    model->sampler->get_token_banned(), model->sampler->get_presence_penalty(),
                    model->sampler->get_frequency_penalty(), model->sampler->get_penalty_decay());
            }
            decoded_idx_batch = model->sampler->sample_batch(logits, model->backend->get_num_vocab(), model->backend->get_num_vocab(), batch_size);
        }

        for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
            model->response_buffer_eos_found_batch[batch_idx] = (decoded_idx_batch[batch_idx] == stop_code);
            decoded_text_batch[batch_idx] = model->tokenizer->decode(decoded_idx_batch[batch_idx]);
            if (!model->response_buffer_eos_found_batch[batch_idx]) {
                model->response_buffer_batch[batch_idx] += decoded_text_batch[batch_idx];
                model->response_buffer_ids_batch[batch_idx].push_back(decoded_idx_batch[batch_idx]);
            }
        }

        int ret = eval_logits_batch_decode(model_id, decoded_idx_batch, logits);
        if (ret) {
            model->is_generating = false;
            LOGE("failed to eval logits\n");
            return ret;
        }

        if (std::all_of(model->response_buffer_eos_found_batch.begin(), model->response_buffer_eos_found_batch.end(), [](bool eos_found) { return eos_found; }) || model->stop_signal) {
            break;
        }

        for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
            occurences_batch[batch_idx][decoded_idx_batch[batch_idx]]++;
        }
    }

    model->is_generating = false;
    model->stop_signal = false;
    return RWKV_SUCCESS;
}

int runtime::gen_completion(int model_id, std::string prompt, int max_length, int stop_code, void (*callback)(const char *, const int, const char *)) {
    if (_models.find(model_id) == _models.end()) {
        LOGE("gen_completion: Model ID %d not found", model_id);
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);
    if (model->backend == nullptr || model->tokenizer == nullptr) {
        LOGE("gen_completion: Backend or tokenizer for model ID %d not found", model_id);
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    model->response_buffer = "";
    model->response_buffer_ids.clear();
    model->response_buffer_eos_found = false;
    model->is_generating = true;
    model->stop_signal = false;

    std::vector<int> ids = model->tokenizer->encode(prompt);
    _prefill_progress_start(ids.size());

    float *logits = nullptr;
    int ret = eval_logits(model_id, ids, logits);
    if (ret || !logits) {
        LOGE("gen_completion: Error evaluating logits");
        model->is_generating = false;
        return ret;
    }
    _prefill_progress_finish();

    model->response_buffer = prompt;
    model->response_buffer_ids = ids;
    static int idx = 0;
    for (int i = 0; i < max_length; i++) {
        model->sampler->apply_penalties(logits, model->backend->get_num_vocab());
        idx = model->sampler->sample(logits, model->backend->get_num_vocab());

        model->response_buffer_eos_found = (idx == stop_code);

        std::string next = model->tokenizer->decode(idx);
        model->response_buffer += next;
        model->response_buffer_ids.push_back(idx);
        ret = eval_logits(model_id, idx, logits);
        if (ret) {
            model->is_generating = false;
            LOGE("failed to eval logits\n");
            return ret;
        }
        if (callback) {
            callback(model->response_buffer.c_str(), idx, next.c_str());
        }

        if (model->response_buffer_eos_found || model->stop_signal) {
            break;
        }

        model->sampler->update_occurences(idx);
    }

    model->is_generating = false;
    model->stop_signal = false;
    return RWKV_SUCCESS;
}

double runtime::get_avg_decode_speed(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return 0.0;
    }
    auto &model = _models.at(model_id);
    double speed_from_backend = model->backend->get_decode_speed();
    if (speed_from_backend > 0) {
        return speed_from_backend;
    }

    if (_decode_speed < 0) {
        return 0.0;
    } else {
        return _decode_speed;
    }
}

double runtime::get_avg_prefill_speed(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return 0.0;
    }
    auto &model = _models.at(model_id);
    double speed_from_backend = model->backend->get_prefill_speed();
    if (speed_from_backend > 0) {
        return speed_from_backend;
    }

    if (_prefill_speed < 0) {
        return 0.0;
    } else {
        return _prefill_speed;
    }
}

void runtime::set_sampler_params(int model_id, float temperature, int top_k, float top_p) {
    if (_models.find(model_id) == _models.end()) {
        return;
    }
    auto &model = _models.at(model_id);
    model->sampler->set_temperature(temperature);
    model->sampler->set_top_k(top_k);
    model->sampler->set_top_p(top_p);
}

void runtime::set_is_generating(int model_id, bool is_generating) {
    if (_models.find(model_id) == _models.end()) {
        return;
    }
    auto &model = _models.at(model_id);
    model->is_generating = is_generating;
}

void runtime::set_stop_signal(int model_id, bool stop_signal) {
    if (_models.find(model_id) == _models.end()) {
        return;
    }
    auto &model = _models.at(model_id);
    model->stop_signal = stop_signal;
}

bool runtime::get_stop_signal(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return false;
    }
    auto &model = _models.at(model_id);
    return model->stop_signal;
}

bool runtime::is_generating(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return false;
    }
    auto &model = _models.at(model_id);
    return model->is_generating;
}

float runtime::get_temperature(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return 1.0f;
    }
    auto &model = _models.at(model_id);
    return model->sampler->get_temperature();
}

int runtime::get_top_k(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return 1;
    }
    auto &model = _models.at(model_id);
    return model->sampler->get_top_k();
}

float runtime::get_top_p(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return 1.0f;
    }
    auto &model = _models.at(model_id);
    return model->sampler->get_top_p();
}

void runtime::set_penalty_params(int model_id, float presence_penalty, float frequency_penalty, float penalty_decay) {
    if (_models.find(model_id) == _models.end()) {
        return;
    }
    auto &model = _models.at(model_id);
    model->sampler->set_presence_penalty(presence_penalty);
    model->sampler->set_frequency_penalty(frequency_penalty);
    model->sampler->set_penalty_decay(penalty_decay);
}

float runtime::get_presence_penalty(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return 0.0f;
    }
    auto &model = _models.at(model_id);
    return model->sampler->get_presence_penalty();
}

float runtime::get_frequency_penalty(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return 0.0f;
    }
    auto &model = _models.at(model_id);
    return model->sampler->get_frequency_penalty();
}

float runtime::get_penalty_decay(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return 0.0f;
    }
    auto &model = _models.at(model_id);
    return model->sampler->get_penalty_decay();
}

void runtime::set_user_role(int model_id, std::string role) {
    if (_models.find(model_id) == _models.end()) {
        return;
    }
    auto &model = _models.at(model_id);
    model->user_role = role;
}

void runtime::set_response_role(int model_id, std::string role) {
    if (_models.find(model_id) == _models.end()) {
        return;
    }
    auto &model = _models.at(model_id);
    model->response_role = role;
}

void runtime::set_bos_token(int model_id, std::string token) {
    if (_models.find(model_id) == _models.end()) {
        return;
    }
    auto &model = _models.at(model_id);
    model->bos_token = token;
}

void runtime::set_eos_token(int model_id, std::string token) {
    if (_models.find(model_id) == _models.end()) {
        return;
    }
    auto &model = _models.at(model_id);
    model->eos_token = token;
    model->stop_codes.clear();
    model->stop_codes.push_back(token);
}

std::string runtime::get_thinking_token(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return "";
    }
    auto &model = _models.at(model_id);
    return model->thinking_token;
}

void runtime::set_thinking_token(int model_id, std::string thinking_token) {
    if (_models.find(model_id) == _models.end()) {
        return;
    }
    auto &model = _models.at(model_id);
    model->thinking_token = thinking_token;
}

std::vector<int> runtime::tokenizer_encode(int model_id, std::string text) {
    if (_models.find(model_id) == _models.end()) {
        return {};
    }
    auto &model = _models.at(model_id);
    if (model->tokenizer == nullptr) {
        return {};
    }
    return model->tokenizer->encode(text);
}

std::string runtime::tokenizer_decode(int model_id, std::vector<int> ids) {
    if (_models.find(model_id) == _models.end()) {
        return "";
    }
    auto &model = _models.at(model_id);
    if (model->tokenizer == nullptr) {
        return "";
    }
    return model->tokenizer->decode(ids);
}

std::string runtime::tokenizer_decode(int model_id, int id) {
    if (_models.find(model_id) == _models.end()) {
        return "";
    }
    auto &model = _models.at(model_id);
    if (model->tokenizer == nullptr) {
        return "";
    }
    return model->tokenizer->decode(id);
}

void runtime::clear_response_buffer(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return;
    }
    auto &model = _models.at(model_id);
    model->response_buffer.clear();
    model->response_buffer_ids.clear();
    model->response_buffer_eos_found = false;
    for (int i = 0; i < model->response_buffer_batch.size(); i++) {
        model->response_buffer_batch[i].clear();
        model->response_buffer_ids_batch[i].clear();
        model->response_buffer_eos_found_batch[i] = false;
    }
}

void runtime::backend_set_extra_str(int model_id, std::string str) {
    if (_models.find(model_id) == _models.end()) {
        return;
    }
    auto &model = _models.at(model_id);
    if (model->backend == nullptr) {
        return;
    }
    model->backend->extra_str = str;
}

int runtime::release() {
    _models.clear();
#ifdef ENABLE_LLAMACPP
    if (_embedding) {
        _embedding->release();
        _embedding = nullptr;
    }
#endif
#ifdef ENABLE_TTS
    if (_sparktts) {
        _sparktts = nullptr;
    }
#endif
    return RWKV_SUCCESS;
}

int runtime::get_vocab_size(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return 0;
    }
    auto &model = _models.at(model_id);
    if (model->backend == nullptr) {
        return 0;
    }
    return model->backend->get_num_vocab();
}

std::string runtime::get_response_buffer_content(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return "";
    }
    auto &model = _models.at(model_id);
    return model->response_buffer;
}

const std::vector<int32_t> runtime::get_response_buffer_ids(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return {};
    }
    auto &model = _models.at(model_id);
    return model->response_buffer_ids;
}

bool runtime::get_response_buffer_eos_found(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return false;
    }
    auto &model = _models.at(model_id);
    return model->response_buffer_eos_found;
}

std::vector<std::string> runtime::get_response_buffer_content_batch(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return {};
    }
    auto &model = _models.at(model_id);
    return model->response_buffer_batch;
}

std::vector<std::vector<int32_t>> runtime::get_response_buffer_ids_batch(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return {};
    }
    auto &model = _models.at(model_id);
    return model->response_buffer_ids_batch;
}

std::vector<bool> runtime::get_response_buffer_eos_found_batch(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return {};
    }
    auto &model = _models.at(model_id);
    return model->response_buffer_eos_found_batch;
}

double runtime::get_prefill_progress(int model_id) {
    return _prefill_progress;
}

void runtime::set_token_banned(int model_id, std::vector<int> token_banned) {
    if (_models.find(model_id) == _models.end()) {
        return;
    }
    auto &model = _models.at(model_id);
    if (model->sampler == nullptr) {
        return;
    }
    model->sampler->set_token_banned(token_banned);
}

std::vector<int> runtime::get_loaded_model_ids() {
    std::vector<int> model_ids;
    for (const auto& pair : _models) {
        model_ids.push_back(pair.first);
    }
    return model_ids;
}

std::map<int, std::map<std::string, std::string>> runtime::get_loaded_models_info() {
    std::map<int, std::map<std::string, std::string>> models_info;

    for (const auto& pair : _models) {
        int model_id = pair.first;
        const auto& model = pair.second;

        std::map<std::string, std::string> model_info;
        model_info["model_path"] = model->model_path;
        model_info["backend_name"] = model->backend_name;
        model_info["tokenizer_path"] = model->tokenizer_path;
        model_info["user_role"] = model->user_role;
        model_info["response_role"] = model->response_role;
        model_info["bos_token"] = model->bos_token;
        model_info["eos_token"] = model->eos_token;
        model_info["thinking_token"] = model->thinking_token;
        model_info["is_generating"] = model->is_generating ? "true" : "false";
        model_info["vocab_size"] = model->backend ? std::to_string(model->backend->get_num_vocab()) : "0";

        models_info[model_id] = model_info;
    }

    return models_info;
}

std::string& runtime::get_model_path_by_id(int model_id) {
    static std::string empty_string;
    if (_models.find(model_id) == _models.end()) {
        return empty_string;
    }
    auto &model = _models.at(model_id);
    return model->model_path;
}

} // namespace rwkvmobile
