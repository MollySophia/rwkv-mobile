#ifndef MNN_RWKV_BACKEND_H
#define MNN_RWKV_BACKEND_H

#include "backend.h"

#include <MNN/AutoTime.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/Module.hpp>

namespace rwkvmobile {

class mnn_rwkv_backend : public execution_provider {
public:
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
    MNN::Interpreter *interpreter;
    MNN::Session *session;
    std::vector<MNN::Tensor *> state_tensors;
    std::vector<float> logits_buffer;
};

}

#endif
