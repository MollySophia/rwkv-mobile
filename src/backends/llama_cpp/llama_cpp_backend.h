#ifndef LLAMA_CPP_BACKEND_H
#define LLAMA_CPP_BACKEND_H

#include "backend.h"
#include "llama.h"

namespace rwkvmobile {

class llama_cpp_backend : public execution_provider {
public:
    ~llama_cpp_backend() {
        release_model();
        release();
    }
    int init(void * extra) override;
    int load_model(std::string model_path, void * extra = nullptr) override;
    int eval(int id, Tensor1D & logits) override;
    int eval(std::vector<int> ids, Tensor1D & logits) override;
    int eval_batch(std::vector<std::vector<int>> ids, Tensor1D & logits) override;
    int eval_batch_tokens(const std::vector<int> &ids, Tensor1D & logits) override;
    int eval_with_embeddings(const float *embeddings, int n_tokens, Tensor1D & logits) override;
    bool is_available() override;
    int zero_state() override;
    int get_state(std::any &state) override;
    int set_state(std::any state) override;
    int free_state(std::any state) override;
    int get_state_on_batch_slot(int slot, std::any &state) override;
    int set_state_on_batch_slot(int slot, std::any state) override;
    int zero_state_on_batch_slot(int slot) override;
    int copy_state_between_batch_slots(int src_slot, int dst_slot) override;
    int release_model() override;
    int release() override;
    int load_raw_states(std::vector<std::vector<half_float::half>> states) override;
    int serialize_runtime_state(std::any state, std::vector<uint8_t> &data) override;
    int deserialize_runtime_state(std::vector<uint8_t> &data, std::any &state) override;

    bool embedding_input_force_no_ln0() override { return true; }
private:
#if defined(__ANDROID__)
    static constexpr int kMaxBatchSlots = 1;
#else
    static constexpr int kMaxBatchSlots = 16;
#endif

    struct replayable_state {
        std::vector<uint8_t> pre_last_token_state;
        std::vector<uint8_t> seq_state;
        int last_token = 0;
        bool has_last_token = false;
    };

    void initialize_supported_batch_sizes();
    int get_state_bytes_for_slot(int slot, std::vector<uint8_t> &state_bytes);
    int set_state_bytes_for_slot(int slot, const std::vector<uint8_t> &state_bytes);
    int restore_replayable_state_on_slot(int slot, const replayable_state &state_data);
    int parse_runtime_state(std::any state, replayable_state &state_data);

    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    std::vector<replayable_state> pending_checkpoint_states;
    llama_batch batch_decode = {};
    bool batch_decode_initialized = false;
};

}

#endif
