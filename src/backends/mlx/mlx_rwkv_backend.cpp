#include <fstream>
#include <filesystem>
#include <dlfcn.h>
#include <cstring>

#include "backend.h"
#include "mlx_rwkv_backend.h"
#include "commondef.h"
#include "logger.h"

namespace rwkvmobile {

static void* mlx_dylib_handle = nullptr;
static bool mlx_dylib_loaded = false;
static bool mlx_initialized = false;

typedef void* (*mlx_model_load_fn)(const char*);
typedef void (*mlx_model_release_fn)(void*);
typedef const char* (*mlx_last_error_message_fn)(void);
typedef int (*mlx_model_get_config_fn)(void*, int32_t*, int32_t*, int32_t*, int32_t*);
typedef int (*mlx_model_eval_fn)(void*, const int32_t*, int32_t, float*);
typedef int32_t (*mlx_cache_get_size_fn)(void*);
typedef int32_t (*mlx_cache_read_fn)(void*, void*, int32_t);
typedef int (*mlx_cache_write_fn)(void*, const void*, int32_t);
typedef int (*mlx_initialize_fn)(void);

static mlx_model_load_fn mlx_model_load_ptr = nullptr;
static mlx_model_release_fn mlx_model_release_ptr = nullptr;
static mlx_last_error_message_fn mlx_last_error_message_ptr = nullptr;
static mlx_model_get_config_fn mlx_model_get_config_ptr = nullptr;
static mlx_model_eval_fn mlx_model_eval_ptr = nullptr;
static mlx_cache_get_size_fn mlx_cache_get_size_ptr = nullptr;
static mlx_cache_read_fn mlx_cache_read_ptr = nullptr;
static mlx_cache_write_fn mlx_cache_write_ptr = nullptr;
static mlx_initialize_fn mlx_initialize_ptr = nullptr;

template<typename T>
static T resolve_symbol(void* handle, const char* symbol_name) {
    if (!handle) {
        return nullptr;
    }
    void* symbol = dlsym(handle, symbol_name);
    if (!symbol) {
        const char* error = dlerror();
        LOGE("Failed to resolve symbol %s: %s\n", symbol_name, error ? error : "Unknown error");
        return nullptr;
    }
    return reinterpret_cast<T>(symbol);
}

int mlx_rwkv_backend::init(void * extra) {
    if (mlx_dylib_loaded && mlx_dylib_handle) {
        LOGI("MLX dynamic library already loaded\n");
        return RWKV_SUCCESS;
    }

    const char* dylib_path;
    if (extra) {
        dylib_path = static_cast<const char*>(extra);
    } else {
        dylib_path = "libMLXModelFFI.dylib";
    }

    mlx_dylib_handle = dlopen(dylib_path, RTLD_LAZY | RTLD_GLOBAL);
    if (!mlx_dylib_handle) {
        const char* error = dlerror();
        LOGE("Failed to load MLX dynamic library: %s\n", error ? error : "Unknown error");
        return RWKV_ERROR_INIT;
    }

    mlx_model_load_ptr = resolve_symbol<mlx_model_load_fn>(mlx_dylib_handle, "mlx_model_load");
    mlx_model_release_ptr = resolve_symbol<mlx_model_release_fn>(mlx_dylib_handle, "mlx_model_release");
    mlx_last_error_message_ptr = resolve_symbol<mlx_last_error_message_fn>(mlx_dylib_handle, "mlx_last_error_message");
    mlx_model_get_config_ptr = resolve_symbol<mlx_model_get_config_fn>(mlx_dylib_handle, "mlx_model_get_config");
    mlx_model_eval_ptr = resolve_symbol<mlx_model_eval_fn>(mlx_dylib_handle, "mlx_model_eval");
    mlx_cache_get_size_ptr = resolve_symbol<mlx_cache_get_size_fn>(mlx_dylib_handle, "mlx_cache_get_size");
    mlx_cache_read_ptr = resolve_symbol<mlx_cache_read_fn>(mlx_dylib_handle, "mlx_cache_read");
    mlx_cache_write_ptr = resolve_symbol<mlx_cache_write_fn>(mlx_dylib_handle, "mlx_cache_write");
    mlx_initialize_ptr = resolve_symbol<mlx_initialize_fn>(mlx_dylib_handle, "mlx_initialize");
    if (!mlx_model_load_ptr || !mlx_model_release_ptr || !mlx_last_error_message_ptr ||
        !mlx_model_get_config_ptr || !mlx_model_eval_ptr || !mlx_cache_get_size_ptr ||
        !mlx_cache_read_ptr || !mlx_cache_write_ptr) {
        if (mlx_dylib_handle) {
            dlclose(mlx_dylib_handle);
            mlx_dylib_handle = nullptr;
        }
        LOGE("Failed to resolve all MLX symbols\n");
        return RWKV_ERROR_INIT;
    }

    mlx_dylib_loaded = true;

    if (!mlx_initialized) {
        if (mlx_initialize_ptr() != 0) {
            LOGE("Failed to initialize MLX\n");
            return RWKV_ERROR_INIT;
        }
        mlx_initialized = true;
    }

    LOGI("MLX dynamic library loaded successfully\n");
    return RWKV_SUCCESS;
}

int mlx_rwkv_backend::load_model(std::string model_path) {
    if (!model_handle && mlx_model_load_ptr) {
        model_handle = mlx_model_load_ptr(model_path.c_str());
        if (!model_handle) {
            LOGE("Failed to load MLX model: %s\n", mlx_last_error_message_ptr());
            return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
        }
    }

    int vocab_size, hidden_size, head_dim, num_layers;
    if (mlx_model_get_config_ptr(model_handle, &vocab_size, &hidden_size, &head_dim, &num_layers) != 0) {
        LOGE("Failed to get MLX model config\n");
        return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
    }

    this->vocab_size = vocab_size;
    this->n_layers = num_layers;
    this->num_heads = head_dim;
    this->hidden_size = hidden_size;

    return RWKV_SUCCESS;
}

int mlx_rwkv_backend::eval(int id, float *& logits) {
    if (!model_handle || !mlx_model_eval_ptr) {
        LOGE("MLX model not loaded\n");
        return RWKV_ERROR_EVAL;
    }

    if (logits_buffer.size() != vocab_size) {
        logits_buffer.resize(vocab_size);
    }

    int ret = mlx_model_eval_ptr(model_handle, &id, 1, logits_buffer.data());
    if (ret != 0) {
        LOGE("Failed to evaluate MLX model: %s\n", mlx_last_error_message_ptr());
        return RWKV_ERROR_EVAL;
    }

    logits = logits_buffer.data();
    return RWKV_SUCCESS;
}

int mlx_rwkv_backend::eval(std::vector<int> ids, float *& logits, bool skip_logits_copy) {
    if (!model_handle || !mlx_model_eval_ptr) {
        LOGE("MLX model not loaded\n");
        return RWKV_ERROR_EVAL;
    }

    if (logits_buffer.size() != vocab_size) {
        logits_buffer.resize(vocab_size);
    }

    std::vector<int32_t> ids_u32(ids.begin(), ids.end());
    int ret = mlx_model_eval_ptr(model_handle, ids_u32.data(), ids_u32.size(), logits_buffer.data());
    if (ret != 0) {
        LOGE("Failed to evaluate MLX model: %s\n", mlx_last_error_message_ptr());
        return RWKV_ERROR_EVAL;
    }

    logits = logits_buffer.data();
    return RWKV_SUCCESS;
}

int mlx_rwkv_backend::get_state(std::any &state) {
    if (!model_handle || !mlx_cache_get_size_ptr) {
        LOGE("MLX model not loaded\n");
        return RWKV_ERROR_EVAL;
    }

    int32_t cache_size = n_layers * (num_heads + 2) * hidden_size * sizeof(__fp16);

    std::vector<__fp16> cache_buffer(cache_size / sizeof(__fp16), 0.0f);
    mlx_cache_read_ptr(model_handle, (void*)cache_buffer.data(), cache_size);
    state = std::move(cache_buffer);

    return RWKV_SUCCESS;
}

int mlx_rwkv_backend::set_state(std::any state) {
    if (!model_handle || !mlx_cache_write_ptr) {
        LOGE("MLX model not loaded\n");
        return RWKV_ERROR_EVAL;
    }

    std::vector<__fp16> cache_buffer = std::any_cast<std::vector<__fp16>>(state);
    int ret = mlx_cache_write_ptr(model_handle, (const void*)cache_buffer.data(), cache_buffer.size() * sizeof(__fp16));
    if (ret != 0) {
        LOGE("Failed to write MLX cache, returned %d, expected %d\n", ret, cache_buffer.size() * sizeof(__fp16));
        return RWKV_ERROR_EVAL;
    }

    return RWKV_SUCCESS;
}

int mlx_rwkv_backend::free_state(std::any state) {
    state.reset();

    return RWKV_SUCCESS;
}

int mlx_rwkv_backend::zero_state() {
    if (!model_handle || !mlx_cache_write_ptr) {
        LOGE("MLX model not loaded\n");
        return RWKV_ERROR_EVAL;
    }

    int32_t cache_size = n_layers * (num_heads + 2) * hidden_size * sizeof(__fp16);
    std::vector<__fp16> cache_buffer(cache_size / sizeof(__fp16), 0.0f);
    int ret = mlx_cache_write_ptr(model_handle, (const void*)cache_buffer.data(), cache_size);
    if (ret != cache_size) {
        LOGE("Failed to write MLX cache\n");
        return RWKV_ERROR_EVAL;
    }

    return RWKV_SUCCESS;
}

int mlx_rwkv_backend::release_model() {
    if (model_handle && mlx_model_release_ptr) {
        mlx_model_release_ptr(model_handle);
        model_handle = NULL;
    }
    return RWKV_SUCCESS;
}

int mlx_rwkv_backend::release() {
    if (model_handle && mlx_model_release_ptr) {
        mlx_model_release_ptr(model_handle);
        model_handle = NULL;
    }
    return RWKV_SUCCESS;
}

bool mlx_rwkv_backend::is_available() {
    return true;
}

} // namespace rwkvmobile