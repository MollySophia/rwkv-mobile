#include "stdint.h"

struct Sampler
{
  float temp;
  float top_p;
  uintptr_t top_k;
};

struct ModelOutput {
  uintptr_t len;
  float *logits;
};

struct ModelOutputBatch {
  uintptr_t batch;
  uintptr_t len;
  float *logits;
};

struct ModelInfoOutput {
  uintptr_t version;
  uintptr_t num_layer;
  uintptr_t num_hidden;
  uintptr_t num_emb;
  uintptr_t num_vocab;
  uintptr_t num_head;
};

struct StateRaw {
  uintptr_t len;
  float *state;
};

#ifdef __cplusplus
extern "C" {
#endif
/// Initialize logger and RNG. Call this once before everything.
void init(uint64_t seed);

/// Set the RNG seed.
void seed(uint64_t seed);

/// Load a runtime.
///
/// # Safety
///
/// The caller must ensure that `model` is valid.
int load(const char *model, uintptr_t quant, uintptr_t quant_nf4, uintptr_t quant_sf4, bool fp16, uintptr_t batch);

typedef void (*load_pth_progress_callback)(float progress);

int load_pth(const char *model, uintptr_t quant, uintptr_t quant_nf4, uintptr_t quant_sf4, bool fp16, uintptr_t batch, load_pth_progress_callback callback);

int load_prefab(const char *model, bool fp16, uintptr_t batch);

int load_extended(const char *model, uintptr_t quant, uintptr_t quant_nf4, uintptr_t quant_sf4, bool fp16, uintptr_t batch);

int load_with_rescale(const char *model, uintptr_t quant, uintptr_t quant_nf4, uintptr_t quant_sf4, uintptr_t rescale, bool fp16, uintptr_t batch);

void release();

/// Clear the model state.
void clear_state(uintptr_t batch);

/// Generate the next token prediction given the input tokens and a sampler.
///
/// # Safety
///
/// The caller must ensure that `tokens` is valid and `len` does not exceed the actual length of `tokens`.
uint32_t infer(const uint32_t *tokens,
               uintptr_t len,
               struct Sampler sampler);

/// Delete the model output vector created by the infer functions.
void free_raw(struct ModelOutput output);

void free_raw_batch(struct ModelOutputBatch output);

/// Compute the model's raw output (next token prediction only) given the input tokens.
///
/// # Safety
///
/// The caller must ensure that `tokens` is valid and `len` does not exceed the actual length of `tokens`.
struct ModelOutput infer_raw_last(const uint32_t *tokens, uintptr_t len);

struct ModelOutputBatch infer_raw_last_batch(const uint32_t **tokens, uintptr_t *len, uintptr_t batch);

/// Compute the model's raw output (predictions of all tokens) given the input tokens.
///
/// # Safety
///
/// The caller must ensure that `tokens` is valid and `len` does not exceed the actual length of `tokens`.
struct ModelOutput infer_raw_all(const uint32_t *tokens, uintptr_t len);

struct ModelOutputBatch infer_raw_all_batch(const uint32_t **tokens, uintptr_t *len, uintptr_t batch);

struct ModelInfoOutput get_model_info();

struct StateRaw get_state(uintptr_t batch);

void set_state(struct StateRaw state, uintptr_t batch);

void free_state(struct StateRaw state);

int convert_pth_to_st(const char *input_path, const char *output_path);

#ifdef __cplusplus
} // extern "C"
#endif