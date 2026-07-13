#ifndef MTK_NP9_BACKEND_H
#define MTK_NP9_BACKEND_H

#include "backend.h"
#include "mtk_rwkv_dlopen.h"
#include "tensor.h"

#include <memory>
#include <string>
#include <vector>

namespace rwkvmobile {

class mtk_np9_backend : public execution_provider {
public:
    ~mtk_np9_backend() {
        release_model();
        release();
    }

    int init(void * extra) override;
    int load_model(std::string model_path, void * extra = nullptr) override;

    int eval(int id, Tensor1D & logits) override;
    int eval(std::vector<int> ids, Tensor1D & logits) override;
    int eval_batch(std::vector<std::vector<int>> ids, Tensor1D & logits) override;
    int eval_with_embeddings(const float *embeddings, int n_tokens, Tensor1D & logits) override;

    bool is_available() override;

    int get_state(std::any &state) override;
    int set_state(std::any state) override;
    int free_state(std::any state) override;
    int zero_state() override;
    int get_state_on_batch_slot(int slot, std::any &state) override;
    int set_state_on_batch_slot(int slot, std::any state) override;
    int zero_state_on_batch_slot(int slot) override;
    int load_raw_states(std::vector<std::vector<half_float::half>> states) override;

    int release_model() override;
    int release() override;
    double get_prefill_speed() override {
        return _prefill_speed;
    }
    double get_decode_speed() override {
        return _decode_speed;
    }
    void reset_speed_stats() override {
        _prefill_speed = -1;
        _decode_speed = -1;
    }

private:
    MtkRwkvDlopen _library;
    void* _runtime = nullptr;
    std::vector<float> _logits_buffer;
    Tensor1D _logits_fp16_view;
    double _prefill_speed = -1;
    double _decode_speed = -1;
    int _prefill_seq_len = 0;
};

} // namespace rwkvmobile

#endif
