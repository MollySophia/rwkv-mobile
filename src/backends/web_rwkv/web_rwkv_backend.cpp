#include <fstream>
#include <filesystem>
#include <cstring>
#include <algorithm>

#include "backend.h"
#include "web_rwkv_backend.h"
#include "commondef.h"
#include "logger.h"
#include <memory>

namespace rwkvmobile {

static thread_local web_rwkv_backend* g_loading_web_rwkv_backend = nullptr;

static void web_rwkv_load_progress_callback(float progress) {
    if (g_loading_web_rwkv_backend) {
        g_loading_web_rwkv_backend->set_load_progress_real(progress);
    }
}

struct web_rwkv_args {
    int quant_type;    // 0: fp, 1: int8, 2: nf4
    int quant_layers;
};

int web_rwkv_backend::init(void * extra) {
    ::init((uint64_t)time(NULL));
    return RWKV_SUCCESS;
}

int web_rwkv_backend::load_model(std::string model_path, void * extra) {
    const int batch_size = 12;
    _load_is_pth = false;
    _load_progress_real = -1.f;

    if (!std::filesystem::exists(model_path)) {
        return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
    }
    bool use_fp16 = true;
    if (model_path.find("respark") != std::string::npos) {
        use_fp16 = false;
    }

    web_rwkv_args *args = nullptr;
    if (extra) {
        args = reinterpret_cast<web_rwkv_args*>(extra);
    }

    int quant = 0;
    int quant_nf4 = 0;
    int quant_sf4 = 0;
    if (args) {
        switch (args->quant_type) {
            case 1:
                quant = args->quant_layers;
                break;
            case 2:
                quant_nf4 = args->quant_layers;
                break;
            case 3:
                quant_sf4 = args->quant_layers;
                break;
            default:
                break;
        }
    }

    int ret = 0;
    if (model_path.find(".pth") != std::string::npos) {
        _load_is_pth = true;
        _load_progress_real = 0.f;
        {
            std::lock_guard<std::mutex> lock(_load_progress_mutex);
            _load_progress_reported = 0.f;
            _load_progress_step = 0.1f;
        }
        g_loading_web_rwkv_backend = this;
        ret = load_pth(model_path.c_str(), quant, quant_nf4, quant_sf4, use_fp16, batch_size, web_rwkv_load_progress_callback);
        g_loading_web_rwkv_backend = nullptr;
        _load_is_pth = false;
    } else if (model_path.find("prefab") != std::string::npos) {
        ret = load_prefab(model_path.c_str(), use_fp16, batch_size);
    } else if (model_path.find("ABC") != std::string::npos
        || model_path.find("abc") != std::string::npos
        || model_path.find("MIDI") != std::string::npos
        || model_path.find("midi") != std::string::npos) {
        ret = load_with_rescale(model_path.c_str(), quant, quant_nf4, quant_sf4, 999, use_fp16, batch_size);
    } else if (model_path.find("extended") != std::string::npos) {
        ret = load_extended(model_path.c_str(), quant, quant_nf4, quant_sf4, use_fp16, batch_size);
    } else { // .st
        ret = load(model_path.c_str(), quant, quant_nf4, quant_sf4, use_fp16, batch_size);
    }
    if (ret != 0) {
        LOGE("web_rwkv_backend::load_model: failed to load model");
        return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
    }

    struct ModelInfoOutput info = get_model_info();
    version = info.version;
    n_layers = info.num_layer;
    num_heads = info.num_head;
    hidden_size = info.num_emb;
    vocab_size = info.num_vocab;

    supported_batch_sizes.clear();
    for (int i = 1; i <= batch_size; i++) {
        supported_batch_sizes.push_back(i);
    }

    return RWKV_SUCCESS;
}

float web_rwkv_backend::get_load_progress() const {
    if (!_load_is_pth.load()) {
        return -1.f;
    }
    float real = _load_progress_real.load();
    if (real < 0.f) {
        return -1.f;
    }
    if (real < 0.5f) {
        return real;
    }
    std::lock_guard<std::mutex> lock(_load_progress_mutex);
    if (_load_progress_reported < real) {
        _load_progress_reported = real;
        _load_progress_step = 0.05f;
    }
    _load_progress_reported = _load_progress_reported + _load_progress_step;
    _load_progress_step = std::max(0.001f, _load_progress_step * 0.9f);
    return std::max(0.f, std::min(1.f, _load_progress_reported));
}

int web_rwkv_backend::eval(int id, Tensor1D & logits) {
    uint32_t id_u32 = (uint32_t)id;
    auto ret = infer_raw_last(&id_u32, 1);
    if (!ret.len || !ret.logits) {
        LOGE("web_rwkv_backend::eval: failed to infer_raw_last");
        return RWKV_ERROR_EVAL;
    }

    if (logits_buffer.size() != vocab_size) {
        logits_buffer.resize(vocab_size);
    }
    memcpy(logits_buffer.data(), ret.logits, vocab_size * sizeof(float));
    logits = Tensor1D::make(logits_buffer.data(), TensorDType::F32, (size_t)vocab_size);

    ::free_raw(ret);
    return RWKV_SUCCESS;
}

int web_rwkv_backend::eval(std::vector<int> ids, Tensor1D & logits) {
    std::vector<uint32_t> ids_u32(ids.begin(), ids.end());
    auto ret = infer_raw_last((const uint32_t *)ids_u32.data(), ids_u32.size());
    if (!ret.len || !ret.logits) {
        return RWKV_ERROR_EVAL;
    }
    if (logits_buffer.size() != vocab_size) {
        logits_buffer.resize(vocab_size);
    }
    memcpy(logits_buffer.data(), ret.logits, vocab_size * sizeof(float));
    logits = Tensor1D::make(logits_buffer.data(), TensorDType::F32, (size_t)vocab_size);

    ::free_raw(ret);
    return RWKV_SUCCESS;
}

int web_rwkv_backend::eval_batch(std::vector<std::vector<int>> ids_batch, Tensor1D & logits) {
    bool supported = false;
    int batch_size = ids_batch.size();
    for (auto b : supported_batch_sizes) {
        if (batch_size == b) {
            supported = true;
            break;
        }
    }
    if (!supported) {
        return RWKV_ERROR_EVAL | RWKV_ERROR_UNSUPPORTED;
    }

    if (logits_buffer.size() != vocab_size * batch_size) {
        logits_buffer.resize(vocab_size * batch_size);
    }

    std::vector<std::vector<uint32_t>> ids_u32_all(ids_batch.size());
    for (int i = 0; i < ids_batch.size(); i++) {
        ids_u32_all[i] = std::vector<uint32_t>(ids_batch[i].begin(), ids_batch[i].end());
    }

    std::vector<uint32_t *> ids_u32_all_ptr(ids_u32_all.size());
    for (int i = 0; i < ids_u32_all.size(); i++) {
        ids_u32_all_ptr[i] = ids_u32_all[i].data();
    }

    std::vector<uintptr_t> len_per_batch;
    for (const auto& ids : ids_batch) {
        len_per_batch.push_back(ids.size());
    }
    auto ret = infer_raw_last_batch((const uint32_t **)ids_u32_all_ptr.data(), len_per_batch.data(), len_per_batch.size());
    if (!ret.len || !ret.logits) {
        return RWKV_ERROR_EVAL;
    }

    for (int i = 0; i < batch_size; i++) {
        memcpy(logits_buffer.data() + i * vocab_size, ret.logits + i * ret.len, vocab_size * sizeof(float));
    }
    logits = Tensor1D::make(logits_buffer.data(), TensorDType::F32, (size_t)(vocab_size * batch_size));

    ::free_raw_batch(ret);
    return RWKV_SUCCESS;
}

bool web_rwkv_backend::is_available() {
    // TODO: Detect this
    return true;
}

int web_rwkv_backend::zero_state() {
    return zero_state_on_batch_slot(0);
}

int web_rwkv_backend::get_state(std::any &state) {
    return get_state_on_batch_slot(0, state);
}

int web_rwkv_backend::set_state(std::any state) {
    return set_state_on_batch_slot(0, state);
}

int web_rwkv_backend::get_state_on_batch_slot(int slot, std::any &state) {
    struct StateRaw raw = ::get_state(slot);
    if (!raw.len || !raw.state) {
        return RWKV_ERROR_BACKEND | RWKV_ERROR_INVALID_PARAMETERS;
    }
    state = std::make_shared<web_rwkv_state>(raw);
    return RWKV_SUCCESS;
}

int web_rwkv_backend::set_state_on_batch_slot(int slot, std::any state) {
    if (!state.has_value()) {
        return RWKV_ERROR_BACKEND | RWKV_ERROR_INVALID_PARAMETERS;
    }
    try {
        const auto& state_ptr = std::any_cast<const std::shared_ptr<web_rwkv_state>&>(state);
        if (!state_ptr) {
            return RWKV_ERROR_BACKEND | RWKV_ERROR_INVALID_PARAMETERS;
        }
        const StateRaw& raw = state_ptr->raw;
        if (!raw.len || !raw.state) {
            return RWKV_ERROR_BACKEND | RWKV_ERROR_INVALID_PARAMETERS;
        }
        ::set_state(raw, slot);
        return RWKV_SUCCESS;
    } catch (const std::bad_any_cast& e) {
        return RWKV_ERROR_BACKEND | RWKV_ERROR_INVALID_PARAMETERS;
    }
}

int web_rwkv_backend::zero_state_on_batch_slot(int slot) {
    ::clear_state(slot);
    return RWKV_SUCCESS;
}

int web_rwkv_backend::free_state(std::any state) {
    if (!state.has_value()) {
        return RWKV_ERROR_BACKEND | RWKV_ERROR_INVALID_PARAMETERS;
    }
    state.reset();
    return RWKV_SUCCESS;
}

int web_rwkv_backend::release_model() {
    ::release();
    return RWKV_SUCCESS;
}

int web_rwkv_backend::release() {
    return RWKV_SUCCESS;
}

int web_rwkv_backend::serialize_runtime_state(std::any state, std::vector<uint8_t> &data) {
    if (!state.has_value()) return RWKV_ERROR_IO;
    auto new_state = std::any_cast<std::shared_ptr<web_rwkv_state>>(state);
    data.insert(data.end(), (uint8_t *)new_state->raw.state, (uint8_t *)new_state->raw.state + new_state->raw.len * sizeof(float));
    return RWKV_SUCCESS;
}

int web_rwkv_backend::deserialize_runtime_state(std::vector<uint8_t> &data, std::any &state) {
    std::any new_state;
    get_state(new_state);
    StateRaw new_state_raw = std::any_cast<std::shared_ptr<web_rwkv_state>>(new_state)->raw;
    if (new_state_raw.len * sizeof(float) != data.size()) {
        LOGE("state size mismatch, expected %d, got %d", new_state_raw.len, data.size());
        return RWKV_ERROR_IO;
    }
    memcpy(new_state_raw.state, data.data(), data.size());
    state = std::move(new_state);
    return RWKV_SUCCESS;
}

} // namespace rwkvmobile