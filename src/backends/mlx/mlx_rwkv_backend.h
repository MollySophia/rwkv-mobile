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
    int load_model(std::string model_path) override;
    int eval(int id, float *& logits) override;
    int eval(std::vector<int> ids, float *& logits, bool skip_logits_copy = false) override;
    bool is_available() override;
    int get_state(std::any &state) override;
    int set_state(std::any state) override;
    int free_state(std::any state) override;
    int zero_state() override;
    int release_model() override;
    int release() override;

private:
    void *model_handle;

    std::vector<float> logits_buffer;
};

}

#endif
