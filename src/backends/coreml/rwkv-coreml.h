#include <stdint.h>
#include <vector>

struct rwkv_coreml_context;

struct rwkv_coreml_context * rwkv_coreml_init(const char * path_model);
void rwkv_coreml_free(struct rwkv_coreml_context * ctx);

void* rwkv_coreml_decode(
        struct rwkv_coreml_context * ctx,
        int token);

void* rwkv_coreml_prefill(
        struct rwkv_coreml_context * ctx,
        std::vector<int> tokens);

int rwkv_coreml_get_vocab_size(struct rwkv_coreml_context * ctx);

int rwkv_coreml_get_n_layers(struct rwkv_coreml_context * ctx);

int rwkv_coreml_get_num_heads(struct rwkv_coreml_context * ctx);

int rwkv_coreml_get_head_dim(struct rwkv_coreml_context * ctx);

int rwkv_coreml_get_hidden_dim(struct rwkv_coreml_context * ctx);

int rwkv_coreml_get_prefill_seq_length(struct rwkv_coreml_context * ctx);

std::vector<std::vector<uint8_t>> rwkv_coreml_get_state(struct rwkv_coreml_context * ctx);

void rwkv_coreml_set_state(struct rwkv_coreml_context * ctx, std::vector<std::vector<uint8_t>> state);

void rwkv_coreml_zero_state(struct rwkv_coreml_context * ctx);
