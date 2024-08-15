#ifndef WEB_RWKV_BACKEND_H
#define WEB_RWKV_BACKEND_H

#include "backend.h"

namespace rwkvmobile {

class web_rwkv_backend : public execution_provider {
public:
    int init(void * extra) override;
    int load_model(std::string model_path) override;
    int eval(int id, std::vector<float> &logits) override;
    int eval(std::vector<int> ids, std::vector<float> &logits) override;
    bool is_available() override;
    // int get_state(std::vector<float> &state) override;
    // int set_state(std::vector<float> state) override;
};

}

#endif