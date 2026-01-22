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
    int eval_with_embeddings(const float *embeddings, int n_tokens, Tensor1D & logits) override;
    bool is_available() override;
    int zero_state() override;
    int get_state(std::any &state) override;
    int set_state(std::any state) override;
    int free_state(std::any state) override;
    int release_model() override;
    int release() override;
    int load_raw_states(std::vector<std::vector<half_float::half>> states) override;
    int serialize_runtime_state(std::any state, std::vector<uint8_t> &data) override;
    int deserialize_runtime_state(std::vector<uint8_t> &data, std::any &state) override;

    bool embedding_input_force_no_ln0() override { return true; }
private:
    llama_model * model;
    llama_context * ctx;
};

}

#endif
