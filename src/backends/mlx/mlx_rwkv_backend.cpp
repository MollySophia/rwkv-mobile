#include "backend.h"
#include "mlx_rwkv_backend.h"
#include "commondef.h"
#include "logger.h"
#include "MLXModelFFI.h"

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

namespace rwkvmobile {

static bool mlx_initialized = false;

#if defined(__APPLE__) && TARGET_OS_IPHONE
static constexpr int kMaxBatchSlots = 4;
#else
static constexpr int kMaxBatchSlots = 16;
#endif

int mlx_rwkv_backend::init(void * extra) {
    if (!mlx_initialized) {
        if (mlx_initialize() != 0) {
            LOGE("Failed to initialize MLX\n");
            return RWKV_ERROR_INIT;
        }
        mlx_initialized = true;
    }

    LOGI("MLX initialized successfully\n");
    return RWKV_SUCCESS;
}

int mlx_rwkv_backend::load_model(std::string model_path, void * extra) {
    if (!model_handle) {
        model_handle = mlx_model_load(model_path.c_str());
        if (!model_handle) {
            LOGE("Failed to load MLX model: %s\n", mlx_last_error_message());
            return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
        }
    }

    int vocab_size, hidden_size, head_dim, num_layers;
    if (mlx_model_get_config(model_handle, &vocab_size, &hidden_size, &head_dim, &num_layers) != 0) {
        LOGE("Failed to get MLX model config\n");
        return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
    }

    this->vocab_size = vocab_size;
    this->n_layers = num_layers;
    this->num_heads = head_dim;
    this->hidden_size = hidden_size;
    supported_batch_sizes.clear();
    supported_batch_sizes.reserve(kMaxBatchSlots);
    for (int i = 1; i <= kMaxBatchSlots; ++i) {
        supported_batch_sizes.push_back(i);
    }

    return RWKV_SUCCESS;
}

int32_t mlx_rwkv_backend::cache_size_bytes() const {
    return n_layers * (num_heads + 2) * hidden_size * (int32_t)sizeof(__fp16);
}

int mlx_rwkv_backend::eval(int id, Tensor1D & logits) {
    if (!model_handle) {
        LOGE("MLX model not loaded\n");
        return RWKV_ERROR_EVAL;
    }

    if (logits_buffer.size() != vocab_size) {
        logits_buffer.resize(vocab_size);
    }

    int ret = mlx_model_eval(model_handle, &id, 1, logits_buffer.data());
    if (ret != 0) {
        LOGE("Failed to evaluate MLX model: %s\n", mlx_last_error_message());
        return RWKV_ERROR_EVAL;
    }

    logits = Tensor1D::make(logits_buffer.data(), TensorDType::F32, (size_t)vocab_size);
    return RWKV_SUCCESS;
}

int mlx_rwkv_backend::eval(std::vector<int> ids, Tensor1D & logits) {
    if (!model_handle) {
        LOGE("MLX model not loaded\n");
        return RWKV_ERROR_EVAL;
    }

    if (logits_buffer.size() != vocab_size) {
        logits_buffer.resize(vocab_size);
    }

    std::vector<int32_t> ids_u32(ids.begin(), ids.end());
    int ret = mlx_model_eval(model_handle, ids_u32.data(), (int32_t)ids_u32.size(), logits_buffer.data());
    if (ret != 0) {
        LOGE("Failed to evaluate MLX model: %s\n", mlx_last_error_message());
        return RWKV_ERROR_EVAL;
    }

    logits = Tensor1D::make(logits_buffer.data(), TensorDType::F32, (size_t)vocab_size);
    return RWKV_SUCCESS;
}

int mlx_rwkv_backend::eval_batch(std::vector<std::vector<int>> ids, Tensor1D & logits) {
    std::vector<int> flat_ids;
    flat_ids.reserve(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i].size() != 1) {
            LOGE("mlx_rwkv_backend::eval_batch only supports single-token decode per slot\n");
            return RWKV_ERROR_UNSUPPORTED;
        }
        flat_ids.push_back(ids[i][0]);
    }
    return eval_batch_tokens(flat_ids, logits);
}

int mlx_rwkv_backend::eval_batch_tokens(const std::vector<int> &ids, Tensor1D & logits) {
    if (!model_handle) {
        LOGE("MLX model not loaded\n");
        return RWKV_ERROR_EVAL;
    }

    const int batch_size = (int)ids.size();
    bool supported = false;
    for (int size : supported_batch_sizes) {
        if (size == batch_size) {
            supported = true;
            break;
        }
    }
    if (!supported) {
        LOGE("MLX batch size %d is not supported\n", batch_size);
        return RWKV_ERROR_EVAL | RWKV_ERROR_UNSUPPORTED;
    }

    const size_t logits_size = (size_t)vocab_size * (size_t)batch_size;
    if (logits_buffer.size() != logits_size) {
        logits_buffer.resize(logits_size);
    }

    std::vector<int32_t> ids_i32(ids.begin(), ids.end());
    int ret = mlx_model_eval_batch_tokens(model_handle, ids_i32.data(), (int32_t)batch_size, logits_buffer.data());
    if (ret != 0) {
        LOGE("Failed to evaluate MLX model batch: %s\n", mlx_last_error_message());
        return RWKV_ERROR_EVAL;
    }

    logits = Tensor1D::make(logits_buffer.data(), TensorDType::F32, logits_size);
    return RWKV_SUCCESS;
}

int mlx_rwkv_backend::get_state(std::any &state) {
    return get_state_on_batch_slot(0, state);
}

int mlx_rwkv_backend::set_state(std::any state) {
    return set_state_on_batch_slot(0, state);
}

int mlx_rwkv_backend::get_state_on_batch_slot(int slot, std::any &state) {
    if (!model_handle) {
        LOGE("MLX model not loaded\n");
        return RWKV_ERROR_EVAL;
    }

    if (slot < 0 || slot >= kMaxBatchSlots) {
        return RWKV_ERROR_BACKEND | RWKV_ERROR_INVALID_PARAMETERS;
    }

    const int32_t cache_size = cache_size_bytes();
    if (cache_size <= 0) {
        return RWKV_ERROR_BACKEND | RWKV_ERROR_INVALID_PARAMETERS;
    }

    std::vector<__fp16> cache_buffer(cache_size / (int32_t)sizeof(__fp16), 0.0f);
    int32_t bytes_read = mlx_cache_read_slot(model_handle, (int32_t)slot, (void*)cache_buffer.data(), cache_size);
    if (bytes_read != cache_size) {
        LOGE("Failed to read MLX cache slot %d: %s\n", slot, mlx_last_error_message() == NULL ? "Unknown error" : mlx_last_error_message());
        return RWKV_ERROR_EVAL;
    }
    state = std::move(cache_buffer);

    return RWKV_SUCCESS;
}

int mlx_rwkv_backend::set_state_on_batch_slot(int slot, std::any state) {
    if (!model_handle) {
        LOGE("MLX model not loaded\n");
        return RWKV_ERROR_EVAL;
    }

    if (slot < 0 || slot >= kMaxBatchSlots) {
        return RWKV_ERROR_BACKEND | RWKV_ERROR_INVALID_PARAMETERS;
    }

    if (!state.has_value()) {
        return zero_state_on_batch_slot(slot);
    }

    try {
        const std::vector<__fp16> &cache_buffer = std::any_cast<const std::vector<__fp16> &>(state);
        const int32_t cache_size = cache_size_bytes();
        if ((int32_t)(cache_buffer.size() * sizeof(__fp16)) != cache_size) {
            LOGE("MLX cache size mismatch, expected %d bytes, got %zu bytes\n", cache_size, cache_buffer.size() * sizeof(__fp16));
            return RWKV_ERROR_BACKEND | RWKV_ERROR_INVALID_PARAMETERS;
        }

        int ret = mlx_cache_write_slot(model_handle, (int32_t)slot, (const void*)cache_buffer.data(), cache_size);
        if (ret != 0) {
            LOGE("Failed to write MLX cache slot %d: %s\n", slot, mlx_last_error_message() == NULL ? "Unknown error" : mlx_last_error_message());
            return RWKV_ERROR_EVAL;
        }
    } catch (const std::bad_any_cast &) {
        return RWKV_ERROR_BACKEND | RWKV_ERROR_INVALID_PARAMETERS;
    }

    return RWKV_SUCCESS;
}

int mlx_rwkv_backend::zero_state() {
    return zero_state_on_batch_slot(0);
}

int mlx_rwkv_backend::zero_state_on_batch_slot(int slot) {
    if (!model_handle) {
        LOGE("MLX model not loaded\n");
        return RWKV_ERROR_EVAL;
    }

    if (slot < 0 || slot >= kMaxBatchSlots) {
        return RWKV_ERROR_BACKEND | RWKV_ERROR_INVALID_PARAMETERS;
    }

    int ret = mlx_cache_zero_slot(model_handle, (int32_t)slot);
    if (ret != 0) {
        LOGE("Failed to zero MLX cache slot %d: %s\n", slot, mlx_last_error_message() == NULL ? "Unknown error" : mlx_last_error_message());
        return RWKV_ERROR_EVAL;
    }

    return RWKV_SUCCESS;
}

int mlx_rwkv_backend::free_state(std::any state) {
    state.reset();

    return RWKV_SUCCESS;
}

int mlx_rwkv_backend::release_model() {
    if (model_handle) {
        mlx_model_release(model_handle);
        model_handle = NULL;
    }
    return RWKV_SUCCESS;
}

int mlx_rwkv_backend::release() {
    return RWKV_SUCCESS;
}

bool mlx_rwkv_backend::is_available() {
    return true;
}

} // namespace rwkvmobile
