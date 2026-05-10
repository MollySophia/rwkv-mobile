#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initializes MLX (must be called before any other MLX operations).
// This ensures the Metal library is properly loaded.
// Returns 0 on success, -1 on failure.
int mlx_initialize(void);

// Loads a model from the given path.
// Returns an opaque handle on success; returns NULL on failure.
void *mlx_model_load(const char *path);

// Releases a handle returned by mlx_model_load. Safe to pass NULL.
void mlx_model_release(void *handle);

// Returns the last error message as a borrowed pointer (do not free).
// The pointer remains valid until the next FFI call that sets a new error.
const char *mlx_last_error_message(void);

// Gets model configuration parameters.
// Returns 0 on success, -1 on failure.
// Output parameters are written to the provided pointers.
int mlx_model_get_config(
    void *handle,
    int32_t *vocab_size,
    int32_t *hidden_size,
    int32_t *head_dim,
    int32_t *num_layers
);

// Evaluates the model with given token IDs.
// ids: array of token IDs
// ids_length: number of token IDs
// logits: output buffer for logits (must be pre-allocated by caller with size >= vocab_size)
// Returns 0 on success, -1 on failure.
int mlx_model_eval(
    void *handle,
    const int32_t *ids,
    int32_t ids_length,
    float *logits
);

// Evaluates one token per batch slot.
// ids: array of token IDs, one per slot
// batch_size: number of active batch slots
// logits: output buffer for logits (must be pre-allocated with size >= batch_size * vocab_size)
// Returns 0 on success, -1 on failure.
int mlx_model_eval_batch_tokens(
    void *handle,
    const int32_t *ids,
    int32_t batch_size,
    float *logits
);

// Gets the total size of cache data in bytes (fp16 format).
// Returns size in bytes on success, -1 on failure.
int32_t mlx_cache_get_size(void *handle);

// Reads cache data into the provided buffer (fp16 format).
// buffer: output buffer for cache data (must be pre-allocated)
// buffer_size: size of the buffer in bytes
// Returns number of bytes read on success, -1 on failure.
int32_t mlx_cache_read(
    void *handle,
    void *buffer,
    int32_t buffer_size
);

// Writes cache data from the provided buffer (fp16 format).
// buffer: input buffer containing cache data
// buffer_size: size of the buffer in bytes
// Returns 0 on success, -1 on failure.
int mlx_cache_write(
    void *handle,
    const void *buffer,
    int32_t buffer_size
);

// Reads a single batch slot from the cache into the provided buffer (fp16 format).
// Returns number of bytes read on success, -1 on failure.
int32_t mlx_cache_read_slot(
    void *handle,
    int32_t slot,
    void *buffer,
    int32_t buffer_size
);

// Writes a single batch slot into the cache from the provided buffer (fp16 format).
// Returns 0 on success, -1 on failure.
int mlx_cache_write_slot(
    void *handle,
    int32_t slot,
    const void *buffer,
    int32_t buffer_size
);

// Clears a single batch slot in the cache.
// Returns 0 on success, -1 on failure.
int mlx_cache_zero_slot(
    void *handle,
    int32_t slot
);

#ifdef __cplusplus
} // extern "C"
#endif

