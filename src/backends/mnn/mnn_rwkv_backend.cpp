#include <fstream>
#include <filesystem>

#include "backend.h"
#include "mnn_rwkv_backend.h"
#include "commondef.h"

#include <MNN/AutoTime.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/Module.hpp>

namespace rwkvmobile {

int mnn_rwkv_backend::init(void * extra) {
    return RWKV_SUCCESS;
}

int mnn_rwkv_backend::load_model(std::string model_path, void * extra) {
    interpreter = MNN::Interpreter::createFromFile(model_path.c_str());
    MNN::ScheduleConfig config;
#ifdef PLATFORM_IS_IOS
    config.type = MNN_FORWARD_NN;
#else
    config.type = MNN_FORWARD_CPU;
#endif
    MNN::BackendConfig backendConfig;
    backendConfig.memory = MNN::BackendConfig::Memory_Low;
    backendConfig.power = MNN::BackendConfig::Power_High;
    backendConfig.precision = MNN::BackendConfig::Precision_Low;
    config.backendConfig = &backendConfig;
    session = interpreter->createSession(config);
    auto inputs = interpreter->getSessionInputAll(session);
    auto outputs = interpreter->getSessionOutputAll(session);
    n_layers = inputs.size() / 3;
    hidden_size = inputs["state0_in"]->height();
    num_heads = inputs["state1_in"]->channel();
    vocab_size = outputs["out"]->height();
    state_tensors.resize(n_layers * 3);
    for (int i = 0; i < n_layers * 3; i++) {
        state_tensors[i] = new MNN::Tensor(inputs["state" + std::to_string(i) + "_in"], MNN::Tensor::CAFFE);
        memset(state_tensors[i]->host<float>(), 0, state_tensors[i]->size());
    }
    return RWKV_SUCCESS;
}

int mnn_rwkv_backend::eval(int id, Tensor1D & logits) {
    int token = id;
    auto inputs = interpreter->getSessionInputAll(session);
    for (int i = 0; i < n_layers * 3; i++) {
        inputs["state" + std::to_string(i) + "_in"]->copyFromHostTensor(state_tensors[i]);
    }
    void* token_input_ptr = inputs["in"]->map(MNN::Tensor::MAP_TENSOR_WRITE, inputs["in"]->getDimensionType());
    memcpy(token_input_ptr, &token, sizeof(int));
    inputs["in"]->unmap(MNN::Tensor::MAP_TENSOR_WRITE,  inputs["in"]->getDimensionType(), token_input_ptr);
    interpreter->runSession(session);
    auto outputs = interpreter->getSessionOutputAll(session);
    void* logits_output_ptr = outputs["out"]->map(MNN::Tensor::MAP_TENSOR_READ, outputs["out"]->getDimensionType());
    if (logits_buffer.size() < vocab_size) {
        logits_buffer.resize(vocab_size);
    }
    memcpy(logits_buffer.data(), logits_output_ptr, vocab_size * sizeof(float));
    outputs["out"]->unmap(MNN::Tensor::MAP_TENSOR_READ, outputs["out"]->getDimensionType(), logits_output_ptr);
    for (int i = 0; i < n_layers * 3; i++) {
        outputs["state" + std::to_string(i) + "_out"]->copyToHostTensor(state_tensors[i]);
    }
    logits = Tensor1D::make(logits_buffer.data(), TensorDType::F32, (size_t)vocab_size);
    return RWKV_SUCCESS;
}

int mnn_rwkv_backend::eval(std::vector<int> ids, Tensor1D & logits) {
    // TODO: sequential prefill
    int ret = RWKV_SUCCESS;
    for (int i = 0; i < ids.size(); i++) {
        ret = eval(ids[i], logits);
        if (ret != RWKV_SUCCESS) {
            return ret;
        }
    }

    return RWKV_SUCCESS;
}

int mnn_rwkv_backend::get_state(std::any &state) {
    // auto new_state = std::vector<ncnn::Mat>(states.size());
    // for (int i = 0; i < states.size(); i++) {
    //     new_state[i] = states[i].clone();
    // }
    // state = new_state;
    return RWKV_SUCCESS;
}

int mnn_rwkv_backend::set_state(std::any state) {
    // auto new_state = std::any_cast<std::vector<ncnn::Mat>>(state);
    // if (new_state.size() != states.size()) {
    //     return RWKV_ERROR_INVALID_PARAMETERS;
    // }
    // for (int i = 0; i < states.size(); i++) {
    //     states[i].clone_from(new_state[i]);
    // }
    return RWKV_SUCCESS;
}

int mnn_rwkv_backend::free_state(std::any state) {
    // auto new_state = std::any_cast<std::vector<ncnn::Mat>>(state);
    // for (auto &mat : new_state) {
    //     mat.release();
    // }
    return RWKV_SUCCESS;
}

int mnn_rwkv_backend::zero_state() {
    // for (auto &state : states) {
    //     state.fill(0.0f);
    // }
    return RWKV_SUCCESS;
}

int mnn_rwkv_backend::release_model() {
    // states.clear();
    // net.clear();
    return RWKV_SUCCESS;
}

int mnn_rwkv_backend::release() {
    return RWKV_SUCCESS;
}

bool mnn_rwkv_backend::is_available() {
    // always available
    return true;
}


} // namespace rwkvmobile