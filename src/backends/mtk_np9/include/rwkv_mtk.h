#pragma once

#include "llm_types.h"

#include <string>
#include <vector>

typedef struct RWKVModelOptions {
    // Sizes
    size_t hiddenSize           = 2048;
    size_t vocabSize            = 65536;
    size_t numLayer             = 24;
} RWKVModelOptions;

typedef struct RWKVRuntimeOptions {
    std::string embPath;
    std::vector<std::string> dlaPathsDecode;
    std::vector<std::string> dlaPathsPrefill;
    std::vector<std::string> sharedWeightsPaths;
    bool useModelBuffers = false;

    // ===== In-memory init (caller-owned; safe to unmap/free after neuron_rwkv_init returns) =====
    // Note: the runtime will deep-copy what it needs during init.
    const void* sharedWeightsBuffer = nullptr;
    size_t sharedWeightsBufferSize = 0;

    // Decode DLA chunks (size must match n_chunks)
    std::vector<const void*> dlaBuffersDecode;
    std::vector<size_t> dlaBufferSizesDecode;
    std::vector<int> dlaBuffersDecodeBatchSizes;
    std::vector<std::vector<const void*>> dlaBuffersDecodeBatch;
    std::vector<std::vector<size_t>> dlaBufferSizesDecodeBatch;

    // RWKV-7 chunking mode: only chunk0 exports v_first; later chunks read chunk0's v_first.
    bool vFirstOutputFirstChunkOnly = false;

    // Prefill DLA chunks (optional; size must match n_chunks if provided)
    std::vector<const void*> dlaBuffersPrefill;
    std::vector<size_t> dlaBufferSizesPrefill;

    // Optional split lmhead DLA. When present, decode/prefill graph output[0] is hidden
    // and lmhead output[0] is logits.
    std::string lmheadPath;
    const void* lmheadBuffer = nullptr;
    size_t lmheadBufferSize = 0;

    // Backward-compatible fields (older buffer-only decode path)
    std::vector<void*> dlaBuffers;
    std::vector<size_t> dlaBufferSizes;

    const void* embBuffer = nullptr;
    size_t embBufferSize = 0;
} RWKVRuntimeOptions;

// ===== Logging (optional) =====
// severity values match internal LogSeverity enum (DEBUG=0..FATAL=4).
typedef void (*neuron_rwkv_log_callback_t)(void* user_data, int severity, const char* tag, const char* msg);
void neuron_rwkv_set_log_callback(neuron_rwkv_log_callback_t cb, void* user_data);

bool neuron_rwkv_init(void** runtime, const RWKVModelOptions& modelOptions,
                       const RWKVRuntimeOptions& runtimeOptions);

void neuron_rwkv_release(void* runtime);

void* neuron_rwkv_inference_once(void* runtime, const int input_token);

extern "C" void* neuron_rwkv_inference_batch(void* runtime, const int* input_tokens, const size_t batch_size);

void* neuron_rwkv_prefill(void* runtime, const int* input_tokens, const size_t num_tokens);

// ===== Embedding-input inference =====
// embeddings layout: contiguous fp32 array of shape (num_tokens, hiddenSize)
// returns: fp16 logits buffer pointer from the last executed graph (same as neuron_rwkv_inference_once/prefill)
void* neuron_rwkv_eval_with_embeddings(void* runtime, const float* embeddings, const size_t num_tokens);

void neuron_rwkv_reset(void* runtime);

// ===== Runtime state IO (input-state only; output->input copy happens after each inference) =====
// Each layer has 3 state tensors in the decode graph: [att, wkv, ffn] in this order.
// These APIs only read/write the decode graph INPUT buffers.
size_t neuron_rwkv_get_att_state_size(void* runtime, const int layer);
size_t neuron_rwkv_get_wkv_state_size(void* runtime, const int layer);
size_t neuron_rwkv_get_ffn_state_size(void* runtime, const int layer);

bool neuron_rwkv_get_att_state(void* runtime, const int layer, void* out, const size_t out_size);
bool neuron_rwkv_get_wkv_state(void* runtime, const int layer, void* out, const size_t out_size);
bool neuron_rwkv_get_ffn_state(void* runtime, const int layer, void* out, const size_t out_size);

bool neuron_rwkv_set_att_state(void* runtime, const int layer, const void* data, const size_t size);
bool neuron_rwkv_set_wkv_state(void* runtime, const int layer, const void* data, const size_t size);
bool neuron_rwkv_set_ffn_state(void* runtime, const int layer, const void* data, const size_t size);

extern "C" bool neuron_rwkv_get_att_state_slot(void* runtime, const int layer, const int slot, void* out, const size_t out_size);
extern "C" bool neuron_rwkv_get_wkv_state_slot(void* runtime, const int layer, const int slot, void* out, const size_t out_size);
extern "C" bool neuron_rwkv_get_ffn_state_slot(void* runtime, const int layer, const int slot, void* out, const size_t out_size);

extern "C" bool neuron_rwkv_set_att_state_slot(void* runtime, const int layer, const int slot, const void* data, const size_t size);
extern "C" bool neuron_rwkv_set_wkv_state_slot(void* runtime, const int layer, const int slot, const void* data, const size_t size);
extern "C" bool neuron_rwkv_set_ffn_state_slot(void* runtime, const int layer, const int slot, const void* data, const size_t size);
extern "C" bool neuron_rwkv_zero_state_slot(void* runtime, const int slot);
