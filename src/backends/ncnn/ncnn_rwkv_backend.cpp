#include <fstream>
#include <filesystem>

#include "backend.h"
#include "ncnn_rwkv_backend.h"
#include "commondef.h"

#include "net.h"
#include "mat.h"

namespace rwkvmobile {

int ncnn_rwkv_backend::init(void * extra) {
    return RWKV_SUCCESS;
}

int ncnn_rwkv_backend::load_model(std::string model_path, void * extra) {
    if (!std::filesystem::exists(model_path)) {
        return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
    }
    auto remove_extension = [](std::string path) {
        size_t lastindex = path.find_last_of(".");
        return path.substr(0, lastindex);
    };
    std::string param_path = remove_extension(model_path) + ".param";
    std::string bin_path = remove_extension(model_path) + ".bin";

    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;

    int ret = 0;
    ret = net.load_param(param_path.c_str());
    if (ret == -1) {
        return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
    }
    ret = net.load_model(bin_path.c_str());
    if (ret == -1) {
        return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
    }

    ncnn::Extractor ex = net.create_extractor();
    ncnn::Mat model_info;
    ex.extract("model_info", model_info);
    version = model_info[0];
    n_layers = model_info[1];
    num_heads = model_info[2];
    int head_size = model_info[3];
    hidden_size = num_heads * head_size;
    vocab_size = model_info[4];

    for (int i = 0; i < n_layers; i++) {
        states.push_back(ncnn::Mat(hidden_size));
        states.push_back(ncnn::Mat(head_size, head_size, num_heads));
        states.push_back(ncnn::Mat(hidden_size));
    }

    for (auto &state : states) {
        state.fill(0.0f);
    }

    return RWKV_SUCCESS;
}

int ncnn_rwkv_backend::eval(int id, Tensor1D & logits) {
    int token = id;
    ncnn::Mat input = ncnn::Mat(1, &token);
    ncnn::Extractor ex = net.create_extractor();

    for (int i = 0; i < n_layers; i++) {
        ex.input(("state_" + std::to_string(3 * i) + "_in").c_str(), states[i * 3]);
        ex.input(("state_" + std::to_string(3 * i + 1) + "_in").c_str(), states[i * 3 + 1]);
        ex.input(("state_" + std::to_string(3 * i + 2) + "_in").c_str(), states[i * 3 + 2]);
    }
    ex.input("token", input);

    for (int i = 0; i < n_layers; i++) {
        ex.extract(("state_" + std::to_string(3 * i) + "_out").c_str(), states[i * 3]);
        ex.extract(("state_" + std::to_string(3 * i + 1) + "_out").c_str(), states[i * 3 + 1]);
        ex.extract(("state_" + std::to_string(3 * i + 2) + "_out").c_str(), states[i * 3 + 2]);
    }

    ex.extract("logits", logits_mat);
    logits = Tensor1D::make((void*)logits_mat.channel(0), TensorDType::F32, (size_t)vocab_size);

    return RWKV_SUCCESS;
}

int ncnn_rwkv_backend::eval(std::vector<int> ids, Tensor1D & logits) {
    // TODO: sequential prefill
    for (int i = 0; i < ids.size(); i++) {
        int id = ids[i];
        ncnn::Mat input = ncnn::Mat(1, &id);
        ncnn::Extractor ex = net.create_extractor();

        for (int i = 0; i < n_layers; i++) {
            ex.input(("state_" + std::to_string(3 * i) + "_in").c_str(), states[i * 3]);
            ex.input(("state_" + std::to_string(3 * i + 1) + "_in").c_str(), states[i * 3 + 1]);
            ex.input(("state_" + std::to_string(3 * i + 2) + "_in").c_str(), states[i * 3 + 2]);
        }
        ex.input("token", input);

        if (i == ids.size() - 1) {
            // Use member logits_mat to keep the underlying buffer alive after returning.
            ex.extract("logits", logits_mat);
            logits = Tensor1D::make((void*)logits_mat.channel(0), TensorDType::F32, (size_t)vocab_size);
        }
        for (int i = 0; i < n_layers; i++) {
            ex.extract(("state_" + std::to_string(3 * i) + "_out").c_str(), states[i * 3]);
            ex.extract(("state_" + std::to_string(3 * i + 1) + "_out").c_str(), states[i * 3 + 1]);
            ex.extract(("state_" + std::to_string(3 * i + 2) + "_out").c_str(), states[i * 3 + 2]);
        }
    }

    return RWKV_SUCCESS;
}

int ncnn_rwkv_backend::get_state(std::any &state) {
    auto new_state = std::vector<ncnn::Mat>(states.size());
    for (int i = 0; i < states.size(); i++) {
        new_state[i] = states[i].clone();
    }
    state = new_state;
    return RWKV_SUCCESS;
}

int ncnn_rwkv_backend::set_state(std::any state) {
    auto new_state = std::any_cast<std::vector<ncnn::Mat>>(state);
    if (new_state.size() != states.size()) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    for (int i = 0; i < states.size(); i++) {
        states[i].clone_from(new_state[i]);
    }
    return RWKV_SUCCESS;
}

int ncnn_rwkv_backend::free_state(std::any state) {
    auto new_state = std::any_cast<std::vector<ncnn::Mat>>(state);
    for (auto &mat : new_state) {
        mat.release();
    }
    return RWKV_SUCCESS;
}

int ncnn_rwkv_backend::zero_state() {
    for (auto &state : states) {
        state.fill(0.0f);
    }
    return RWKV_SUCCESS;
}

int ncnn_rwkv_backend::release_model() {
    states.clear();
    net.clear();
    return RWKV_SUCCESS;
}

int ncnn_rwkv_backend::release() {
    return RWKV_SUCCESS;
}

bool ncnn_rwkv_backend::is_available() {
    // always available
    return true;
}

} // namespace rwkvmobile