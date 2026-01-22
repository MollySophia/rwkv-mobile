#ifndef MTK_NP7_BACKEND_H
#define MTK_NP7_BACKEND_H

#include "backend.h"
#include "tensor.h"

#include <memory>
#include <string>
#include <vector>

namespace rwkvmobile {

class mtk_np7_backend : public execution_provider {
public:
    ~mtk_np7_backend() {
        release_model();
        release();
    }

    int init(void * extra) override;
    int load_model(std::string model_path, void * extra = nullptr) override;

    int eval(int id, Tensor1D & logits) override;
    int eval(std::vector<int> ids, Tensor1D & logits) override;
    int eval_with_embeddings(const float *embeddings, int n_tokens, Tensor1D & logits) override;

    bool is_available() override;

    int get_state(std::any &state) override;
    int set_state(std::any state) override;
    int free_state(std::any state) override;
    int zero_state() override;
    int load_raw_states(std::vector<std::vector<half_float::half>> states) override;

    int release_model() override;
    int release() override;

private:
    void* _runtime = nullptr;
    std::vector<float> _logits_buffer;
    Tensor1D _logits_fp16_view;
};

} // namespace rwkvmobile

#endif


