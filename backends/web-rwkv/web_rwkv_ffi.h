#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <new>

struct Sampler {
  float temp;
  float top_p;
  uintptr_t top_k;
};

extern "C" {

/// Initialize logger and RNG. Call this once before everything.
void web_rwkv_init(uint64_t seed);

/// Set the RNG seed.
void web_rwkv_seed(uint64_t seed);

/// Load a runtime.
///
/// # Safety
///
/// The caller must ensure that `model` is valid.
void web_rwkv_load(const char *model, uintptr_t quant, uintptr_t quant_nf4);

void web_rwkv_load_with_rescale(const char *model, uintptr_t quant, uintptr_t quant_nf4, uintptr_t rescale);

/// Clear the model state.
void web_rwkv_clear_state();

/// Generate the next token prediction given the input tokens and a sampler.
///
/// # Safety
///
/// The caller must ensure that `tokens` is valid and `len` does not exceed the actual length of `tokens`.
uint16_t web_rwkv_infer(const uint16_t *tokens,
               uintptr_t len,
               Sampler sampler);

int32_t web_rwkv_infer_logits(const uint16_t *tokens,
                    uintptr_t len,
                    float *probs,
                    uintptr_t probs_len);

} // extern "C"