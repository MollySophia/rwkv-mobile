#ifndef NCNN_RWKV_BACKEND_H
#define NCNN_RWKV_BACKEND_H

#include "backend.h"
#include "net.h"
#include "mat.h"

namespace rwkvmobile {

class ncnn_rwkv_backend : public execution_provider {
public:
    ~ncnn_rwkv_backend() {
        release_model();
        release();
    }
    int init(void * extra) override;
    int load_model(std::string model_path, void * extra = nullptr) override;
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
    ncnn::Net net;
    std::vector<ncnn::Mat> states;
    ncnn::Mat logits_mat;
};

}

#endif
