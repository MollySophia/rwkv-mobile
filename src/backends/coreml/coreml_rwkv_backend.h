#ifndef COREML_RWKV_BACKEND_H
#define COREML_RWKV_BACKEND_H

#include "backend.h"
#include "rwkv-coreml.h"

namespace rwkvmobile {

class coreml_rwkv_backend : public execution_provider {
public:
    ~coreml_rwkv_backend() {
        release_model();
        release();
    }
    int init(void * extra) override;
    int load_model(std::string model_path) override;
    int eval(int id, Tensor1D & logits) override;
    int eval(std::vector<int> ids, Tensor1D & logits) override;
    bool is_available() override;
    int get_state(std::any &state) override;
    int set_state(std::any state) override;
    int free_state(std::any state) override;
    int zero_state() override;
    int release_model() override;
    int release() override;

private:
    rwkv_coreml_context * ctx;
};

}

#endif
