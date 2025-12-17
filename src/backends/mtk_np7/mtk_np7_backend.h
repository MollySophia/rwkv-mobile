#ifndef MTK_NP7_BACKEND_H
#define MTK_NP7_BACKEND_H

#include "backend.h"

#include <memory>
#include <string>
#include <vector>

namespace rwkvmobile {

// MediaTek NP7 backend (prebuilt librwkv_mtk.a + public headers only).
class mtk_np7_backend : public execution_provider {
public:
    ~mtk_np7_backend() {
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
    int load_raw_states(std::vector<std::vector<half_float::half>> states) override;

    int release_model() override;
    int release() override;

private:
    void* _runtime = nullptr;
    std::vector<float> _logits_buffer;
};

} // namespace rwkvmobile

#endif


