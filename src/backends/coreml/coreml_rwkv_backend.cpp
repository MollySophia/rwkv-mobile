#include "backend.h"
#include "coreml_rwkv_backend.h"
#include "commondef.h"
#include "logger.h"
#include "rwkv-coreml.h"

namespace rwkvmobile {

int coreml_rwkv_backend::init(void * extra) {
    return RWKV_SUCCESS;
}

int coreml_rwkv_backend::load_model(std::string model_path, void * extra) {
    if (ctx) {
        rwkv_coreml_free(ctx);
        ctx = nullptr;
    }
    ctx = rwkv_coreml_new_context();
    if (ctx == nullptr) {
        return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
    }
    if (rwkv_coreml_init(ctx, model_path.c_str()) != 0) {
        rwkv_coreml_free(ctx);
        ctx = nullptr;
        return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
    }

    vocab_size = rwkv_coreml_get_vocab_size(ctx);
    n_layers = rwkv_coreml_get_n_layers(ctx);
    num_heads = rwkv_coreml_get_num_heads(ctx);
    hidden_size = rwkv_coreml_get_hidden_dim(ctx);
    prefill_seq_length = rwkv_coreml_get_prefill_seq_length(ctx);

    return RWKV_SUCCESS;
}

float coreml_rwkv_backend::get_load_progress() const {
    if (!ctx) {
        return 1.0f;
    }
    return rwkv_coreml_get_load_progress(ctx);
}

int coreml_rwkv_backend::eval(int id, Tensor1D & logits) {
    void* logits_ptr = rwkv_coreml_decode(ctx, id);
    logits = Tensor1D::make(logits_ptr, TensorDType::F16, (size_t)vocab_size);
    return RWKV_SUCCESS;
}

int coreml_rwkv_backend::eval(std::vector<int> ids, Tensor1D & logits) {
    int i = 0;
    for (; i + prefill_seq_length <= ids.size(); i += prefill_seq_length) {
        std::vector<int> tokens_to_prefill = std::vector<int>(ids.begin() + i, ids.begin() + i + prefill_seq_length);
        void* logits_ptr = rwkv_coreml_prefill(ctx, tokens_to_prefill);
        logits = Tensor1D::make(logits_ptr, TensorDType::F16, (size_t)vocab_size);
    }
    for (; i < ids.size(); i++) {
        int ret = eval(ids[i], logits);
        if (ret != RWKV_SUCCESS) {
            return ret;
        }
    }

    return RWKV_SUCCESS;
}

int coreml_rwkv_backend::get_state(std::any &state) {
    std::vector<std::vector<uint8_t>> state_vec = rwkv_coreml_get_state(ctx);
    state = state_vec;
    return RWKV_SUCCESS;
}

int coreml_rwkv_backend::set_state(std::any state) {
    std::vector<std::vector<uint8_t>> state_vec = std::any_cast<std::vector<std::vector<uint8_t>>>(state);
    rwkv_coreml_set_state(ctx, state_vec);
    return RWKV_SUCCESS;
}

int coreml_rwkv_backend::free_state(std::any state) {
    std::vector<std::vector<uint8_t>> state_vec = std::any_cast<std::vector<std::vector<uint8_t>>>(state);
    for (int i = 0; i < state_vec.size(); i++) {
        state_vec[i].clear();
    }
    state_vec.clear();
    return RWKV_SUCCESS;
}

int coreml_rwkv_backend::load_raw_states(std::vector<std::vector<half_float::half>> states) {
    rwkv_coreml_zero_state(ctx);
    std::vector<half_float::half> wkv_state(states.size() * states[0].size());
    for (int i = 0; i < states.size(); i++) {
        memcpy(wkv_state.data() + i * states[i].size(), states[i].data(), states[i].size() * sizeof(half_float::half));
    }
    rwkv_coreml_set_wkv_state(ctx, wkv_state);
    return RWKV_SUCCESS;
}

int coreml_rwkv_backend::serialize_runtime_state(std::any state, std::vector<uint8_t> &data) {
    if (!state.has_value()) return RWKV_ERROR_IO;
    auto new_state = std::any_cast<std::vector<std::vector<uint8_t>>>(state);
    data = new_state[0];
    data.insert(data.end(), new_state[1].begin(), new_state[1].end());
    return RWKV_SUCCESS;
}

int coreml_rwkv_backend::deserialize_runtime_state(std::vector<uint8_t> &data, std::any &state) {
    std::vector<std::vector<uint8_t>> new_state(2);
    auto state_wkv_bytes = rwkv_coreml_get_state_wkv_bytes(ctx);
    new_state[0] = std::vector<uint8_t>(data.begin(), data.begin() + state_wkv_bytes);
    new_state[1] = std::vector<uint8_t>(data.begin() + state_wkv_bytes, data.end());
    state = std::any(new_state);
    return RWKV_SUCCESS;
}

int coreml_rwkv_backend::zero_state() {
    rwkv_coreml_zero_state(ctx);
    return RWKV_SUCCESS;
}

int coreml_rwkv_backend::release_model() {
    if (ctx) {
        rwkv_coreml_free(ctx);
        ctx = NULL;
    }
    return RWKV_SUCCESS;
}

int coreml_rwkv_backend::release() {
    if (ctx) {
        rwkv_coreml_free(ctx);
        ctx = NULL;
    }
    return RWKV_SUCCESS;
}

bool coreml_rwkv_backend::is_available() {
    return true;
}

} // namespace rwkvmobile