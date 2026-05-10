#ifndef MLX_RWKV_BACKEND_H
#define MLX_RWKV_BACKEND_H

#include "backend.h"
#include "MLXModelFFI.h"

namespace rwkvmobile {

class mlx_rwkv_backend : public execution_provider {
public:
    ~mlx_rwkv_backend() {
        release_model();
        release();
    }
    int init(void * extra) override;
    int load_model(std::string model_path, void * extra = nullptr) override;
    int eval(int id, Tensor1D & logits) override;
    int eval(std::vector<int> ids, Tensor1D & logits) override;
    int eval_batch(std::vector<std::vector<int>> ids, Tensor1D & logits) override;
    int eval_batch_tokens(const std::vector<int> &ids, Tensor1D & logits) override;
    bool is_available() override;
    int get_state(std::any &state) override;
    int set_state(std::any state) override;
    int free_state(std::any state) override;
    int zero_state() override;
    int get_state_on_batch_slot(int slot, std::any &state) override;
    int set_state_on_batch_slot(int slot, std::any state) override;
    int zero_state_on_batch_slot(int slot) override;
    int release_model() override;
    int release() override;

private:
    int32_t cache_size_bytes() const;

    void *model_handle = nullptr;

    std::vector<float> logits_buffer;
};

}

#endif
