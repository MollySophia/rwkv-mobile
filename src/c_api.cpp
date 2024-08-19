#include "runtime.h"
#include "commondef.h"
#include "c_api.h"


namespace rwkvmobile {

extern "C" {

rwkvmobile_runtime_t rwkvmobile_runtime_init_with_name(const char * backend_name) {
    runtime * rt = new runtime();
    if (rt == nullptr) {
        return nullptr;
    }
    rt->init(backend_name);
    return rt;
}

int rwkvmobile_runtime_load_model(rwkvmobile_runtime_t handle, const char * model_path) {
    if (handle == nullptr || model_path == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class runtime *>(handle);
    return rt->load_model(model_path);
}

int rwkvmobile_runtime_load_tokenizer(rwkvmobile_runtime_t handle, const char * vocab_file) {
    if (handle == nullptr || vocab_file == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class runtime *>(handle);
    return rt->load_tokenizer(vocab_file);
}

int rwkvmobile_runtime_eval_logits(rwkvmobile_runtime_t handle, const int * ids, int ids_len, float * logits, int logits_len) {
    if (handle == nullptr || ids == nullptr || logits == nullptr || ids_len <= 0 || logits_len <= 0) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class runtime *>(handle);
    std::vector<int> ids_vec(ids, ids + ids_len);
    std::vector<float> logits_vec(logits, logits + logits_len);
    return rt->eval_logits(ids_vec, logits_vec);
}

int rwkvmobile_runtime_eval_chat(
    rwkvmobile_runtime_t handle,
    const char * user_role,
    const char * response_role,
    const char * user_input,
    char * response,
    const int max_length) {
    if (handle == nullptr || user_role == nullptr || response_role == nullptr || user_input == nullptr || response == nullptr || max_length <= 0) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }

    auto rt = static_cast<class runtime *>(handle);
    std::string response_str;
    int ret = rt->chat(
        std::string(user_role),
        std::string(response_role),
        std::string(user_input),
        response_str,
        max_length);
    if (ret != RWKV_SUCCESS) {
        return ret;
    }
    strncpy(response, response_str.c_str(), response_str.size());
    return RWKV_SUCCESS;
}

int rwkvmobile_runtime_gen_completion(
    rwkvmobile_runtime_t handle,
    const char * prompt,
    char * completion,
    const int length) {
    if (handle == nullptr || prompt == nullptr || completion == nullptr || length <= 0) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }

    auto rt = static_cast<class runtime *>(handle);
    std::string completion_str;
    int ret = rt->gen_completion(
        std::string(prompt),
        completion_str,
        length);
    if (ret != RWKV_SUCCESS) {
        return ret;
    }
    strncpy(completion, completion_str.c_str(), completion_str.size());
    return RWKV_SUCCESS;
}

int rwkvmobile_runtime_clear_state(rwkvmobile_runtime_t handle) {
    if (handle == nullptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto rt = static_cast<class runtime *>(handle);
    return rt->clear_state();
}

} // extern "C"
} // namespace rwkvmobile
