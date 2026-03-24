#include <fstream>
#include <filesystem>
#include <thread>

#include "backend.h"
#include "llama_cpp_backend.h"
#include "llama.h"
#include "llama-model.h"
#include "llama-memory-recurrent.h"
#include "commondef.h"
#include "logger.h"

namespace rwkvmobile {

int llama_cpp_backend::init(void * extra) {
    llama_log_set([](enum ggml_log_level level, const char * text, void * /* user_data */) {
        std::string log_msg = std::string(text);
        if (log_msg.empty()) {
            return;
        }
        while (log_msg.size() > 0 && log_msg[log_msg.size() - 1] == '\n') {
            log_msg = log_msg.substr(0, log_msg.size() - 1);
        }
        switch (level) {
            case GGML_LOG_LEVEL_ERROR:
                LOGE("%s", log_msg.c_str());
                break;
            case GGML_LOG_LEVEL_WARN:
                LOGW("%s", log_msg.c_str());
                break;
            case GGML_LOG_LEVEL_INFO:
                LOGI("%s", log_msg.c_str());
                break;
            case GGML_LOG_LEVEL_DEBUG:
                LOGD("%s", log_msg.c_str());
                break;
            default:
                break;
        }
    }, nullptr);

    return RWKV_SUCCESS;
}

int llama_cpp_backend::load_model(std::string model_path, void * extra) {
    llama_model_params model_params = llama_model_default_params();

#if defined(__APPLE__) || defined(__MACH__) || defined(GGML_USE_VULKAN)
    model_params.n_gpu_layers = 99;
#else
    model_params.n_gpu_layers = 0;
#endif
    model_params.progress_callback = nullptr;

    LOGI("n_gpu_layers: %d", model_params.n_gpu_layers);
    model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 1048576;
    ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
    }

// #ifdef __ANDROID__
//     // TODO: set according to the number of prime cores on the device
//     llama_set_n_threads(ctx, 2, 2);
// #endif

    vocab_size = model->vocab.n_tokens();
    hidden_size = llama_model_n_embd(model);
    num_heads = hidden_size / 64;
    n_layers = llama_model_n_layer(model);
    return RWKV_SUCCESS;
}

int llama_cpp_backend::eval(int id, Tensor1D & logits) {
    llama_batch batch = llama_batch_get_one(&id, 1);
    llama_decode(ctx, batch);

    float * logits_out = llama_get_logits_ith(ctx, -1);
    if (!logits_out) {
        return RWKV_ERROR_EVAL;
    }
    logits = Tensor1D::make((void*)logits_out, TensorDType::F32, (size_t)vocab_size);

    return RWKV_SUCCESS;
}

int llama_cpp_backend::eval(std::vector<int> ids, Tensor1D & logits) {
    llama_batch batch = llama_batch_get_one(ids.data(), ids.size());
    llama_decode(ctx, batch);
    float * logits_out = llama_get_logits_ith(ctx, -1);
    if (!logits_out) {
        return RWKV_ERROR_EVAL;
    }
    logits = Tensor1D::make((void*)logits_out, TensorDType::F32, (size_t)vocab_size);

    return RWKV_SUCCESS;
}

int llama_cpp_backend::eval_with_embeddings(const float *embeddings, int n_tokens, Tensor1D & logits) {
    int n_embd = llama_model_n_embd(model);

    llama_batch batch = {
        /*n_tokens       =*/ n_tokens,
        /*tokens         =*/ nullptr,
        /*embd           =*/ (float *)embeddings,
        /*pos            =*/ nullptr,
        /*n_seq_id       =*/ nullptr,
        /*seq_id         =*/ nullptr,
        /*logits         =*/ nullptr,
    };
    llama_decode(ctx, batch);
    float * logits_out = llama_get_logits_ith(ctx, -1);
    if (!logits_out) {
        return RWKV_ERROR_EVAL;
    }
    logits = Tensor1D::make((void*)logits_out, TensorDType::F32, (size_t)vocab_size);

    return RWKV_SUCCESS;
}

bool llama_cpp_backend::is_available() {
    return true;
}

int llama_cpp_backend::zero_state() {
    llama_memory_clear(llama_get_memory(ctx), true);
    return RWKV_SUCCESS;
}

int llama_cpp_backend::get_state(std::any &state) {
    std::vector<uint8_t> state_mem(llama_state_get_size(ctx));
    llama_state_get_data(ctx, state_mem.data(), state_mem.size());
    state = std::move(state_mem);
    return RWKV_SUCCESS;
}

int llama_cpp_backend::set_state(std::any state) {
    try {
        std::vector<uint8_t> state_mem = std::any_cast<std::vector<uint8_t>>(state);
        llama_state_set_data(ctx, state_mem.data(), state_mem.size());
    } catch (const std::bad_any_cast &e) {
        return RWKV_ERROR_BACKEND | RWKV_ERROR_INVALID_PARAMETERS;
    }
    return RWKV_SUCCESS;
}

int llama_cpp_backend::free_state(std::any state) {
    try {
        std::vector<uint8_t> state_mem = std::any_cast<std::vector<uint8_t>>(state);
        state_mem.clear();
    } catch (const std::bad_any_cast &e) {
        return RWKV_ERROR_BACKEND | RWKV_ERROR_INVALID_PARAMETERS;
    }
    return RWKV_SUCCESS;
}

int llama_cpp_backend::load_raw_states(std::vector<std::vector<half_float::half>> states) {
    zero_state();
    Tensor1D logits;
    eval(0, logits);
    llama_memory_recurrent * mem = (llama_memory_recurrent *)llama_get_memory(ctx);
    for (int i = 0; i < n_layers; i++) {
        ggml_tensor * r = mem->r_l[i];
        ggml_tensor * s = mem->s_l[i];

        if (s->ne[0] != hidden_size * (hidden_size / num_heads)) {
            LOGE("state size mismatch, expected %d, got %d", hidden_size * (hidden_size / num_heads), s->ne[0]);
            return RWKV_ERROR_BACKEND | RWKV_ERROR_INVALID_PARAMETERS;
        }

        std::vector<float> state_f32(s->ne[0]);
        for (int j = 0; j < s->ne[0]; j++) {
            state_f32[j] = states[i][j];
        }
        ggml_backend_tensor_set(s, state_f32.data(), 0, state_f32.size() * sizeof(float));
        ggml_backend_tensor_memset(r, 0, 0, r->ne[0] * sizeof(float));
    }

    return RWKV_SUCCESS;
}

int llama_cpp_backend::serialize_runtime_state(std::any state, std::vector<uint8_t> &data) {
    if (!state.has_value()) return RWKV_ERROR_IO;
    auto new_state = std::any_cast<std::vector<uint8_t>>(state);
    data = std::vector<uint8_t>(new_state);
    return RWKV_SUCCESS;
}

int llama_cpp_backend::deserialize_runtime_state(std::vector<uint8_t> &data, std::any &state) {
    state = std::any(std::vector<uint8_t>(data));
    return RWKV_SUCCESS;
}

int llama_cpp_backend::release_model() {
    if (ctx)
        llama_free(ctx);
    if (model)
        llama_model_free(model);
    return RWKV_SUCCESS;
}

int llama_cpp_backend::release() {
    return RWKV_SUCCESS;
}

} // namespace rwkvmobile
