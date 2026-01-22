#include "mtk_np7_backend.h"

#include "commondef.h"
#include "logger.h"
#include "rmpack.h"

#include "rwkv_mtk.h"

#include <filesystem>
#include <cstdint>
#include <stdexcept>

namespace rwkvmobile {

namespace {

struct LoadedRMPackModel {
    RWKVModelOptions modelOptions{};
    RWKVRuntimeOptions runtimeOptions{};
    int n_chunks = 1;
    int num_heads = 0;
    bool use_shared_weights = false;
    bool has_prefill = false;
    std::unique_ptr<RMPackReader> reader;

    void unmapAfterInit() {
        if (!reader) return;
        reader->unmapFile("embedding");
        if (use_shared_weights) {
            reader->unmapFile("shared_weights");
        }
        for (int i = 0; i < n_chunks; ++i) {
            reader->unmapFile("decode_chunk" + std::to_string(i));
            if (has_prefill) {
                reader->unmapFile("prefill_chunk" + std::to_string(i));
            }
        }
    }
};

static void requireFile(RMPackReader& reader, const std::string& name) {
    if (!reader.hasFile(name)) {
        throw std::runtime_error("rmpack missing file: " + name);
    }
}

static LoadedRMPackModel loadFromRMPack(const std::string& rmpackPath) {
    LoadedRMPackModel out;
    out.reader = std::make_unique<RMPackReader>(rmpackPath);

    const auto& cfg = out.reader->getConfig();

    out.modelOptions.hiddenSize = cfg.value("hidden_size", (int)out.modelOptions.hiddenSize);
    out.modelOptions.vocabSize = cfg.value("vocab_size", (int)out.modelOptions.vocabSize);
    out.modelOptions.numLayer  = cfg.value("n_layer", (int)out.modelOptions.numLayer);
    int head_size = cfg.value("head_size", 0);
    if (head_size == 0) {
        throw std::runtime_error("head_size is not set in rmpack config");
    }
    out.num_heads = (int)(out.modelOptions.hiddenSize / head_size);

    out.n_chunks = cfg.value("n_chunks", 1);
    out.use_shared_weights = (cfg.value("use_shared_weights", 0) != 0);

    out.runtimeOptions.useModelBuffers = true;

    // embedding
    requireFile(*out.reader, "embedding");
    out.runtimeOptions.embBuffer = out.reader->mmapFile("embedding");
    out.runtimeOptions.embBufferSize = out.reader->getFileSize("embedding");

    // shared weights (optional)
    if (out.use_shared_weights) {
        requireFile(*out.reader, "shared_weights");
        out.runtimeOptions.sharedWeightsBuffer = out.reader->mmapFile("shared_weights");
        out.runtimeOptions.sharedWeightsBufferSize = out.reader->getFileSize("shared_weights");
    }

    // decode chunks
    out.runtimeOptions.dlaBuffersDecode.reserve(out.n_chunks);
    out.runtimeOptions.dlaBufferSizesDecode.reserve(out.n_chunks);
    for (int i = 0; i < out.n_chunks; ++i) {
        const std::string name = "decode_chunk" + std::to_string(i);
        requireFile(*out.reader, name);
        out.runtimeOptions.dlaBuffersDecode.push_back(out.reader->mmapFile(name));
        out.runtimeOptions.dlaBufferSizesDecode.push_back(out.reader->getFileSize(name));
    }

    // prefill chunks (optional): only enable if all chunks exist
    out.has_prefill = true;
    for (int i = 0; i < out.n_chunks; ++i) {
        const std::string name = "prefill_chunk" + std::to_string(i);
        if (!out.reader->hasFile(name)) {
            out.has_prefill = false;
            break;
        }
    }

    if (out.has_prefill) {
        out.runtimeOptions.dlaBuffersPrefill.reserve(out.n_chunks);
        out.runtimeOptions.dlaBufferSizesPrefill.reserve(out.n_chunks);
        for (int i = 0; i < out.n_chunks; ++i) {
            const std::string name = "prefill_chunk" + std::to_string(i);
            out.runtimeOptions.dlaBuffersPrefill.push_back(out.reader->mmapFile(name));
            out.runtimeOptions.dlaBufferSizesPrefill.push_back(out.reader->getFileSize(name));
        }
    }

    return out;
}

static void mtk_np7_librwkv_mtk_log_cb(void* /*user_data*/, int severity, const char* tag, const char* msg) {
    const char* safe_tag = tag ? tag : "librwkv_mtk";
    const char* safe_msg = msg ? msg : "";
    switch (severity) {
        case 0: // DEBUG
            LOGD("[mtk_np7][%s] %s", safe_tag, safe_msg);
            break;
        case 1: // INFO
            LOGI("[mtk_np7][%s] %s", safe_tag, safe_msg);
            break;
        case 2: // WARN
            LOGW("[mtk_np7][%s] %s", safe_tag, safe_msg);
            break;
        case 3: // ERROR
            LOGE("[mtk_np7][%s] %s", safe_tag, safe_msg);
            break;
        case 4: // FATAL
        default:
            LOGE("[mtk_np7][%s] %s", safe_tag, safe_msg);
            break;
    }
}

} // namespace

int mtk_np7_backend::init(void * extra) {
    (void)extra;
    // Route librwkv_mtk logs through rwkv-mobile logger.
    neuron_rwkv_set_log_callback(mtk_np7_librwkv_mtk_log_cb, nullptr);
    return RWKV_SUCCESS;
}

int mtk_np7_backend::load_model(std::string model_path, void * extra) {
    if (!std::filesystem::exists(model_path)) {
        return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
    }

    // Clean any existing runtime first.
    release_model();

    LoadedRMPackModel loaded;
    try {
        loaded = loadFromRMPack(model_path);
    } catch (const std::exception& e) {
        LOGE("[mtk_np7] Failed to load rmpack: %s\n", e.what());
        return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
    }

    // Ensure callback is set before runtime init so init-time logs are captured.
    neuron_rwkv_set_log_callback(mtk_np7_librwkv_mtk_log_cb, nullptr);

    if (!neuron_rwkv_init(&_runtime, loaded.modelOptions, loaded.runtimeOptions)) {
        LOGE("[mtk_np7] neuron_rwkv_init failed\n");
        loaded.unmapAfterInit();
        _runtime = nullptr;
        return RWKV_ERROR_INIT | RWKV_ERROR_BACKEND;
    }

    // Safe to unmap after init completes (runtime deep-copies what it needs).
    loaded.unmapAfterInit();

    // Expose model info to runtime.
    hidden_size = (int)loaded.modelOptions.hiddenSize;
    vocab_size  = (int)loaded.modelOptions.vocabSize;
    n_layers    = (int)loaded.modelOptions.numLayer;

    version     = 7;
    num_heads   = loaded.num_heads;
    supported_batch_sizes = {1};

    _logits_buffer.resize(vocab_size);

    neuron_rwkv_reset(_runtime);
    return RWKV_SUCCESS;
}

int mtk_np7_backend::eval(int id, Tensor1D & logits) {
    if (_runtime == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    void* logits_ptr = neuron_rwkv_inference_once(_runtime, id);
    if (!logits_ptr) {
        return RWKV_ERROR_EVAL | RWKV_ERROR_BACKEND;
    }

    // RWKV MTK runtime returns fp16 logits buffer.
    _logits_fp16_view = Tensor1D::make(logits_ptr, TensorDType::F16, (size_t)vocab_size);

    // Prefer returning fp16 logits to avoid an expensive full-vocab conversion.
    // Callers that require fp32 can convert on-demand (e.g. before sampling).
    logits = _logits_fp16_view;
    return RWKV_SUCCESS;
}

int mtk_np7_backend::eval(std::vector<int> ids, Tensor1D & logits) {
    if (_runtime == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    if (ids.empty()) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }

    void* logits_ptr = neuron_rwkv_prefill(_runtime, ids.data(), ids.size());
    if (!logits_ptr) {
        return RWKV_ERROR_EVAL | RWKV_ERROR_BACKEND;
    }

    _logits_fp16_view = Tensor1D::make(logits_ptr, TensorDType::F16, (size_t)vocab_size);
    logits = _logits_fp16_view;
    return RWKV_SUCCESS;
}

int mtk_np7_backend::eval_with_embeddings(const float *embeddings, int n_tokens, Tensor1D & logits) {
    if (_runtime == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    if (embeddings == nullptr || n_tokens <= 0) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }

    void* logits_ptr = neuron_rwkv_eval_with_embeddings(_runtime, embeddings, (size_t)n_tokens);
    if (!logits_ptr) {
        return RWKV_ERROR_EVAL | RWKV_ERROR_BACKEND;
    }

    _logits_fp16_view = Tensor1D::make(logits_ptr, TensorDType::F16, (size_t)vocab_size);
    logits = _logits_fp16_view;
    return RWKV_SUCCESS;
}

bool mtk_np7_backend::is_available() {
    return true;
}

int mtk_np7_backend::get_state(std::any &state) {
    if (_runtime == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    // states.size() = 3 * n_layers, order: [att, wkv, ffn] per layer.
    auto states_ptr = std::make_shared<std::vector<std::vector<uint8_t>>>();
    states_ptr->resize((size_t)n_layers * 3);

    for (int layer = 0; layer < n_layers; ++layer) {
        const size_t att_sz = neuron_rwkv_get_att_state_size(_runtime, layer);
        const size_t wkv_sz = neuron_rwkv_get_wkv_state_size(_runtime, layer);
        const size_t ffn_sz = neuron_rwkv_get_ffn_state_size(_runtime, layer);
        if (att_sz == 0 || wkv_sz == 0 || ffn_sz == 0) {
            return RWKV_ERROR_BACKEND | RWKV_ERROR_RUNTIME;
        }

        auto& att = (*states_ptr)[(size_t)layer * 3 + 0];
        auto& wkv = (*states_ptr)[(size_t)layer * 3 + 1];
        auto& ffn = (*states_ptr)[(size_t)layer * 3 + 2];

        att.resize(att_sz);
        wkv.resize(wkv_sz);
        ffn.resize(ffn_sz);

        if (!neuron_rwkv_get_att_state(_runtime, layer, att.data(), att.size())) return RWKV_ERROR_BACKEND | RWKV_ERROR_RUNTIME;
        if (!neuron_rwkv_get_wkv_state(_runtime, layer, wkv.data(), wkv.size())) return RWKV_ERROR_BACKEND | RWKV_ERROR_RUNTIME;
        if (!neuron_rwkv_get_ffn_state(_runtime, layer, ffn.data(), ffn.size())) return RWKV_ERROR_BACKEND | RWKV_ERROR_RUNTIME;
    }

    state = states_ptr;
    return RWKV_SUCCESS;
}

int mtk_np7_backend::set_state(std::any state) {
    if (_runtime == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    if (!state.has_value()) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }

    std::shared_ptr<std::vector<std::vector<uint8_t>>> states_ptr;
    try {
        states_ptr = std::any_cast<std::shared_ptr<std::vector<std::vector<uint8_t>>>>(state);
    } catch (const std::bad_any_cast&) {
        // Allow passing by value as well.
        try {
            auto by_value = std::any_cast<std::vector<std::vector<uint8_t>>>(state);
            states_ptr = std::make_shared<std::vector<std::vector<uint8_t>>>(std::move(by_value));
        } catch (const std::bad_any_cast&) {
            return RWKV_ERROR_INVALID_PARAMETERS;
        }
    }

    if (!states_ptr) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    if ((int)states_ptr->size() != 3 * n_layers) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }

    for (int layer = 0; layer < n_layers; ++layer) {
        const auto& att = (*states_ptr)[(size_t)layer * 3 + 0];
        const auto& wkv = (*states_ptr)[(size_t)layer * 3 + 1];
        const auto& ffn = (*states_ptr)[(size_t)layer * 3 + 2];

        if (!neuron_rwkv_set_att_state(_runtime, layer, att.data(), att.size())) return RWKV_ERROR_BACKEND | RWKV_ERROR_RUNTIME;
        if (!neuron_rwkv_set_wkv_state(_runtime, layer, wkv.data(), wkv.size())) return RWKV_ERROR_BACKEND | RWKV_ERROR_RUNTIME;
        if (!neuron_rwkv_set_ffn_state(_runtime, layer, ffn.data(), ffn.size())) return RWKV_ERROR_BACKEND | RWKV_ERROR_RUNTIME;
    }
    return RWKV_SUCCESS;
}

int mtk_np7_backend::free_state(std::any state) {
    state.reset();
    return RWKV_SUCCESS;
}

int mtk_np7_backend::zero_state() {
    if (_runtime == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    neuron_rwkv_reset(_runtime);
    return RWKV_SUCCESS;
}

int mtk_np7_backend::load_raw_states(std::vector<std::vector<half_float::half>> states) {
    // Used by rwkv-mobile rmpack state loader (one file per layer).
    // Interpret it as WKV state per layer, and zero ATTN/FFN states.
    if (_runtime == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    if ((int)states.size() != n_layers) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }

    const size_t att_sz = neuron_rwkv_get_att_state_size(_runtime, 0);
    const size_t wkv_sz = neuron_rwkv_get_wkv_state_size(_runtime, 0);
    const size_t ffn_sz = neuron_rwkv_get_ffn_state_size(_runtime, 0);
    if (att_sz == 0 || wkv_sz == 0 || ffn_sz == 0) {
        return RWKV_ERROR_BACKEND | RWKV_ERROR_RUNTIME;
    }

    for (int layer = 0; layer < n_layers; ++layer) {
        // Zero att/ffn.
        std::vector<uint8_t> zeros(att_sz, 0);
        if (!neuron_rwkv_set_att_state(_runtime, layer, zeros.data(), zeros.size())) return RWKV_ERROR_BACKEND | RWKV_ERROR_RUNTIME;
        zeros.assign(ffn_sz, 0);
        if (!neuron_rwkv_set_ffn_state(_runtime, layer, zeros.data(), zeros.size())) return RWKV_ERROR_BACKEND | RWKV_ERROR_RUNTIME;

        // Load wkv.
        const auto& wkv_half = states[layer];
        const size_t bytes = wkv_half.size() * sizeof(half_float::half);
        if (bytes != wkv_sz) {
            LOGE("[mtk_np7] load_raw_states: layer %d size mismatch: got=%zu, want=%zu\n", layer, bytes, wkv_sz);
            return RWKV_ERROR_INVALID_PARAMETERS;
        }
        if (!neuron_rwkv_set_wkv_state(_runtime, layer, wkv_half.data(), bytes)) return RWKV_ERROR_BACKEND | RWKV_ERROR_RUNTIME;
    }

    return RWKV_SUCCESS;
}

int mtk_np7_backend::release_model() {
    if (_runtime) {
        neuron_rwkv_release(_runtime);
        _runtime = nullptr;
    }
    return RWKV_SUCCESS;
}

int mtk_np7_backend::release() {
    return RWKV_SUCCESS;
}

} // namespace rwkvmobile


