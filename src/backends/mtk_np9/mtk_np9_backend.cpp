#include "mtk_np9_backend.h"

#include "commondef.h"
#include "logger.h"
#include "rmpack.h"

#include "include/rwkv_mtk.h"

#include <filesystem>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace rwkvmobile {

namespace {

struct TypedMtkRwkvApi {
    using SetLogCallbackFn = void (*)(neuron_rwkv_log_callback_t cb, void* user_data);
    using InitFn = bool (*)(void** runtime, const RWKVModelOptions& modelOptions,
                            const RWKVRuntimeOptions& runtimeOptions);
    using ReleaseFn = void (*)(void* runtime);
    using InferenceOnceFn = void* (*)(void* runtime, int input_token);
    using InferenceBatchFn = void* (*)(void* runtime, const int* input_tokens, size_t batch_size);
    using PrefillFn = void* (*)(void* runtime, const int* input_tokens, size_t num_tokens);
    using EvalWithEmbeddingsFn = void* (*)(void* runtime, const float* embeddings, size_t num_tokens);
    using ResetFn = void (*)(void* runtime);
    using GetStateSizeFn = size_t (*)(void* runtime, int layer);
    using GetStateFn = bool (*)(void* runtime, int layer, void* out, size_t out_size);
    using SetStateFn = bool (*)(void* runtime, int layer, const void* data, size_t size);
    using GetStateSlotFn = bool (*)(void* runtime, int layer, int slot, void* out, size_t out_size);
    using SetStateSlotFn = bool (*)(void* runtime, int layer, int slot, const void* data, size_t size);
    using ZeroStateSlotFn = bool (*)(void* runtime, int slot);

    SetLogCallbackFn set_log_callback = nullptr;
    InitFn init = nullptr;
    ReleaseFn release = nullptr;
    InferenceOnceFn inference_once = nullptr;
    InferenceBatchFn inference_batch = nullptr;
    PrefillFn prefill = nullptr;
    EvalWithEmbeddingsFn eval_with_embeddings = nullptr;
    ResetFn reset = nullptr;
    GetStateSizeFn get_att_state_size = nullptr;
    GetStateSizeFn get_wkv_state_size = nullptr;
    GetStateSizeFn get_ffn_state_size = nullptr;
    GetStateFn get_att_state = nullptr;
    GetStateFn get_wkv_state = nullptr;
    GetStateFn get_ffn_state = nullptr;
    SetStateFn set_att_state = nullptr;
    SetStateFn set_wkv_state = nullptr;
    SetStateFn set_ffn_state = nullptr;
    GetStateSlotFn get_att_state_slot = nullptr;
    GetStateSlotFn get_wkv_state_slot = nullptr;
    GetStateSlotFn get_ffn_state_slot = nullptr;
    SetStateSlotFn set_att_state_slot = nullptr;
    SetStateSlotFn set_wkv_state_slot = nullptr;
    SetStateSlotFn set_ffn_state_slot = nullptr;
    ZeroStateSlotFn zero_state_slot = nullptr;
};

template <typename Fn>
static Fn cast_symbol(void* symbol) {
    return reinterpret_cast<Fn>(symbol);
}

static TypedMtkRwkvApi mtk_api(MtkRwkvDlopen& library) {
    const auto& raw = library.api();
    return {
        cast_symbol<TypedMtkRwkvApi::SetLogCallbackFn>(raw.set_log_callback),
        cast_symbol<TypedMtkRwkvApi::InitFn>(raw.init),
        cast_symbol<TypedMtkRwkvApi::ReleaseFn>(raw.release),
        cast_symbol<TypedMtkRwkvApi::InferenceOnceFn>(raw.inference_once),
        cast_symbol<TypedMtkRwkvApi::InferenceBatchFn>(raw.inference_batch),
        cast_symbol<TypedMtkRwkvApi::PrefillFn>(raw.prefill),
        cast_symbol<TypedMtkRwkvApi::EvalWithEmbeddingsFn>(raw.eval_with_embeddings),
        cast_symbol<TypedMtkRwkvApi::ResetFn>(raw.reset),
        cast_symbol<TypedMtkRwkvApi::GetStateSizeFn>(raw.get_att_state_size),
        cast_symbol<TypedMtkRwkvApi::GetStateSizeFn>(raw.get_wkv_state_size),
        cast_symbol<TypedMtkRwkvApi::GetStateSizeFn>(raw.get_ffn_state_size),
        cast_symbol<TypedMtkRwkvApi::GetStateFn>(raw.get_att_state),
        cast_symbol<TypedMtkRwkvApi::GetStateFn>(raw.get_wkv_state),
        cast_symbol<TypedMtkRwkvApi::GetStateFn>(raw.get_ffn_state),
        cast_symbol<TypedMtkRwkvApi::SetStateFn>(raw.set_att_state),
        cast_symbol<TypedMtkRwkvApi::SetStateFn>(raw.set_wkv_state),
        cast_symbol<TypedMtkRwkvApi::SetStateFn>(raw.set_ffn_state),
        cast_symbol<TypedMtkRwkvApi::GetStateSlotFn>(raw.get_att_state_slot),
        cast_symbol<TypedMtkRwkvApi::GetStateSlotFn>(raw.get_wkv_state_slot),
        cast_symbol<TypedMtkRwkvApi::GetStateSlotFn>(raw.get_ffn_state_slot),
        cast_symbol<TypedMtkRwkvApi::SetStateSlotFn>(raw.set_att_state_slot),
        cast_symbol<TypedMtkRwkvApi::SetStateSlotFn>(raw.set_wkv_state_slot),
        cast_symbol<TypedMtkRwkvApi::SetStateSlotFn>(raw.set_ffn_state_slot),
        cast_symbol<TypedMtkRwkvApi::ZeroStateSlotFn>(raw.zero_state_slot),
    };
}

static bool is_hot_path_sdk_debug_log(const char* tag, const char* msg) {
    if (!tag || !msg) {
        return false;
    }
    if (std::strcmp(tag, "llm_sdk_latency") == 0) {
        return std::strstr(msg, "runInferenceImpl:") != nullptr;
    }
    if (std::strcmp(tag, "llm_sdk") != 0) {
        return false;
    }
    return std::strstr(msg, "[requiresInit] done") != nullptr ||
           std::strstr(msg, "[runInferenceImpl] done") != nullptr;
}

struct LoadedRMPackModel {
    RWKVModelOptions modelOptions{};
    RWKVRuntimeOptions runtimeOptions{};
    int n_chunks = 1;
    int prefill_seq_len = 0;
    int num_heads = 0;
    bool use_shared_weights = false;
    bool has_prefill = false;
    std::vector<int> decode_batch_sizes;
    std::unique_ptr<RMPackReader> reader;

    void unmapAfterInit() {
        if (!reader) return;
        reader->unmapFile("embedding");
        if (use_shared_weights) {
            reader->unmapFile("shared_weights");
        }
        if (reader->hasFile("lmhead")) {
            reader->unmapFile("lmhead");
        }
        for (int i = 0; i < n_chunks; ++i) {
            reader->unmapFile("decode_chunk" + std::to_string(i));
            for (int bsz : decode_batch_sizes) {
                reader->unmapFile("decode_bsz" + std::to_string(bsz) + "_chunk" + std::to_string(i));
            }
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
    out.prefill_seq_len = cfg.value("prefill_seq_len", 0);
    out.use_shared_weights = (cfg.value("use_shared_weights", 0) != 0);
    out.decode_batch_sizes = cfg.value("decode_batch_sizes", std::vector<int>{});
    std::sort(out.decode_batch_sizes.begin(), out.decode_batch_sizes.end());
    out.decode_batch_sizes.erase(std::unique(out.decode_batch_sizes.begin(), out.decode_batch_sizes.end()), out.decode_batch_sizes.end());

    out.runtimeOptions.useModelBuffers = true;
    out.runtimeOptions.vFirstOutputFirstChunkOnly =
        (cfg.value("v_first_output_first_chunk_only", 0) != 0);

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

    for (int bsz : out.decode_batch_sizes) {
        if (bsz <= 1) {
            continue;
        }
        bool has_all_chunks = true;
        for (int i = 0; i < out.n_chunks; ++i) {
            const std::string name = "decode_bsz" + std::to_string(bsz) + "_chunk" + std::to_string(i);
            if (!out.reader->hasFile(name)) {
                has_all_chunks = false;
                break;
            }
        }
        if (!has_all_chunks) {
            throw std::runtime_error("rmpack missing complete decode batch graph for bsz" + std::to_string(bsz));
        }
        out.runtimeOptions.dlaBuffersDecodeBatchSizes.push_back(bsz);
        out.runtimeOptions.dlaBuffersDecodeBatch.emplace_back();
        out.runtimeOptions.dlaBufferSizesDecodeBatch.emplace_back();
        auto& buffers = out.runtimeOptions.dlaBuffersDecodeBatch.back();
        auto& sizes = out.runtimeOptions.dlaBufferSizesDecodeBatch.back();
        buffers.reserve(out.n_chunks);
        sizes.reserve(out.n_chunks);
        for (int i = 0; i < out.n_chunks; ++i) {
            const std::string name = "decode_bsz" + std::to_string(bsz) + "_chunk" + std::to_string(i);
            buffers.push_back(out.reader->mmapFile(name));
            sizes.push_back(out.reader->getFileSize(name));
        }
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

    if (out.reader->hasFile("lmhead")) {
        out.runtimeOptions.lmheadBuffer = out.reader->mmapFile("lmhead");
        out.runtimeOptions.lmheadBufferSize = out.reader->getFileSize("lmhead");
    }

    return out;
}

static void mtk_np9_librwkv_mtk_log_cb(void* /*user_data*/, int severity, const char* tag, const char* msg) {
    const char* safe_tag = tag ? tag : "librwkv_mtk";
    const char* safe_msg = msg ? msg : "";
    if (severity == 0 && is_hot_path_sdk_debug_log(tag, msg)) {
        return;
    }
    switch (severity) {
        case 0: // DEBUG
            LOGD("[mtk_np9][%s] %s", safe_tag, safe_msg);
            break;
        case 1: // INFO
            LOGI("[mtk_np9][%s] %s", safe_tag, safe_msg);
            break;
        case 2: // WARN
            LOGW("[mtk_np9][%s] %s", safe_tag, safe_msg);
            break;
        case 3: // ERROR
            LOGE("[mtk_np9][%s] %s", safe_tag, safe_msg);
            break;
        case 4: // FATAL
        default:
            LOGE("[mtk_np9][%s] %s", safe_tag, safe_msg);
            break;
    }
}

} // namespace

int mtk_np9_backend::init(void * extra) {
    const int ret = _library.open("mtk_np9", "RWKV_MTK_NP9_LIB", "librwkv_mtk_np9.so", extra);
    if (ret != RWKV_SUCCESS) {
        return ret;
    }
    // Route librwkv_mtk logs through rwkv-mobile logger.
    mtk_api(_library).set_log_callback(mtk_np9_librwkv_mtk_log_cb, nullptr);
    return RWKV_SUCCESS;
}

int mtk_np9_backend::load_model(std::string model_path, void * extra) {
    if (!std::filesystem::exists(model_path)) {
        return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
    }

    // Clean any existing runtime first.
    release_model();

    try {
        LoadedRMPackModel loaded = loadFromRMPack(model_path);

        // Ensure callback is set before runtime init so init-time logs are captured.
        mtk_api(_library).set_log_callback(mtk_np9_librwkv_mtk_log_cb, nullptr);

        if (!mtk_api(_library).init(&_runtime, loaded.modelOptions, loaded.runtimeOptions)) {
            LOGE("[mtk_np9] neuron_rwkv_init failed\n");
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
        _prefill_seq_len = loaded.has_prefill ? loaded.prefill_seq_len : 0;

        version     = 7;
        num_heads   = loaded.num_heads;
        supported_batch_sizes = {1};
        if (mtk_api(_library).inference_batch != nullptr &&
            mtk_api(_library).get_att_state_slot != nullptr &&
            mtk_api(_library).set_att_state_slot != nullptr) {
            for (int bsz : loaded.decode_batch_sizes) {
                if (bsz > 1) {
                    for (int i = 2; i <= bsz; ++i) {
                        supported_batch_sizes.push_back(i);
                    }
                }
            }
            std::sort(supported_batch_sizes.begin(), supported_batch_sizes.end());
            supported_batch_sizes.erase(std::unique(supported_batch_sizes.begin(), supported_batch_sizes.end()), supported_batch_sizes.end());
        }
    } catch (const std::exception& e) {
        LOGE("[mtk_np9] Failed to load rmpack: %s\n", e.what());
        return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
    }

    _logits_buffer.resize(vocab_size);

    mtk_api(_library).reset(_runtime);
    return RWKV_SUCCESS;
}

int mtk_np9_backend::eval(int id, Tensor1D & logits) {
    if (_runtime == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto start = std::chrono::high_resolution_clock::now();
    void* logits_ptr = mtk_api(_library).inference_once(_runtime, id);
    auto end = std::chrono::high_resolution_clock::now();
    if (!logits_ptr) {
        return RWKV_ERROR_EVAL | RWKV_ERROR_BACKEND;
    }
    const int64_t duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    if (duration_us > 0) {
        _decode_speed = 1000000.0 / (double)duration_us;
    }

    // RWKV MTK runtime returns fp16 logits buffer.
    _logits_fp16_view = Tensor1D::make(logits_ptr, TensorDType::F16, (size_t)vocab_size);

    // Prefer returning fp16 logits to avoid an expensive full-vocab conversion.
    // Callers that require fp32 can convert on-demand (e.g. before sampling).
    logits = _logits_fp16_view;
    return RWKV_SUCCESS;
}

int mtk_np9_backend::eval(std::vector<int> ids, Tensor1D & logits) {
    if (_runtime == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    if (ids.empty()) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }

    auto api = mtk_api(_library);
    void* logits_ptr = nullptr;
    const size_t full_tokens = (_prefill_seq_len > 1)
        ? (ids.size() / (size_t)_prefill_seq_len) * (size_t)_prefill_seq_len
        : 0;

    if (full_tokens > 0) {
        auto start = std::chrono::high_resolution_clock::now();
        logits_ptr = api.prefill(_runtime, ids.data(), full_tokens);
        auto end = std::chrono::high_resolution_clock::now();
        if (!logits_ptr) {
            return RWKV_ERROR_EVAL | RWKV_ERROR_BACKEND;
        }
        const int64_t duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        if (duration_us > 0) {
            _prefill_speed = (double)full_tokens * 1000000.0 / (double)duration_us;
        }
    }

    int64_t decode_duration_us = 0;
    int decode_tokens = 0;
    for (size_t i = full_tokens; i < ids.size(); ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        logits_ptr = api.inference_once(_runtime, ids[i]);
        auto end = std::chrono::high_resolution_clock::now();
        if (!logits_ptr) {
            return RWKV_ERROR_EVAL | RWKV_ERROR_BACKEND;
        }
        decode_duration_us += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        decode_tokens++;
    }

    if (!logits_ptr) {
        return RWKV_ERROR_EVAL | RWKV_ERROR_BACKEND;
    }
    if (decode_tokens > 0 && decode_duration_us > 0) {
        _decode_speed = (double)decode_tokens * 1000000.0 / (double)decode_duration_us;
    }

    _logits_fp16_view = Tensor1D::make(logits_ptr, TensorDType::F16, (size_t)vocab_size);
    logits = _logits_fp16_view;
    return RWKV_SUCCESS;
}

int mtk_np9_backend::eval_batch(std::vector<std::vector<int>> ids, Tensor1D & logits) {
    if (_runtime == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    if (ids.empty()) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    for (const auto& seq : ids) {
        if (seq.size() != 1) {
            return RWKV_ERROR_UNSUPPORTED;
        }
    }
    auto api = mtk_api(_library);
    if (api.inference_batch == nullptr) {
        return RWKV_ERROR_UNSUPPORTED;
    }

    std::vector<int> token_ids(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        token_ids[i] = ids[i][0];
    }

    auto start = std::chrono::high_resolution_clock::now();
    void* logits_ptr = api.inference_batch(_runtime, token_ids.data(), token_ids.size());
    auto end = std::chrono::high_resolution_clock::now();
    if (!logits_ptr) {
        return RWKV_ERROR_EVAL | RWKV_ERROR_BACKEND;
    }
    const int64_t duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    if (duration_us > 0) {
        _decode_speed = (double)ids.size() * 1000000.0 / (double)duration_us;
    }

    _logits_fp16_view = Tensor1D::make(logits_ptr, TensorDType::F16, (size_t)vocab_size * ids.size());
    logits = _logits_fp16_view;
    return RWKV_SUCCESS;
}

int mtk_np9_backend::eval_with_embeddings(const float *embeddings, int n_tokens, Tensor1D & logits) {
    if (_runtime == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    if (embeddings == nullptr || n_tokens <= 0) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }

    auto api = mtk_api(_library);
    void* logits_ptr = nullptr;
    const int full_tokens = (_prefill_seq_len > 1)
        ? (n_tokens / _prefill_seq_len) * _prefill_seq_len
        : 0;

    if (full_tokens > 0) {
        auto start = std::chrono::high_resolution_clock::now();
        logits_ptr = api.eval_with_embeddings(_runtime, embeddings, (size_t)full_tokens);
        auto end = std::chrono::high_resolution_clock::now();
        if (!logits_ptr) {
            return RWKV_ERROR_EVAL | RWKV_ERROR_BACKEND;
        }
        const int64_t duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        if (duration_us > 0) {
            _prefill_speed = (double)full_tokens * 1000000.0 / (double)duration_us;
        }
    }

    int64_t decode_duration_us = 0;
    int decode_tokens = 0;
    for (int i = full_tokens; i < n_tokens; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        logits_ptr = api.eval_with_embeddings(
            _runtime,
            embeddings + (size_t)i * (size_t)hidden_size,
            1
        );
        auto end = std::chrono::high_resolution_clock::now();
        if (!logits_ptr) {
            return RWKV_ERROR_EVAL | RWKV_ERROR_BACKEND;
        }
        decode_duration_us += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        decode_tokens++;
    }

    if (!logits_ptr) {
        return RWKV_ERROR_EVAL | RWKV_ERROR_BACKEND;
    }
    if (decode_tokens > 0 && decode_duration_us > 0) {
        _decode_speed = (double)decode_tokens * 1000000.0 / (double)decode_duration_us;
    }

    _logits_fp16_view = Tensor1D::make(logits_ptr, TensorDType::F16, (size_t)vocab_size);
    logits = _logits_fp16_view;
    return RWKV_SUCCESS;
}

bool mtk_np9_backend::is_available() {
    return true;
}

int mtk_np9_backend::get_state(std::any &state) {
    if (_runtime == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    // states.size() = 3 * n_layers, order: [att, wkv, ffn] per layer.
    auto states_ptr = std::make_shared<std::vector<std::vector<uint8_t>>>();
    states_ptr->resize((size_t)n_layers * 3);

    for (int layer = 0; layer < n_layers; ++layer) {
        const size_t att_sz = mtk_api(_library).get_att_state_size(_runtime, layer);
        const size_t wkv_sz = mtk_api(_library).get_wkv_state_size(_runtime, layer);
        const size_t ffn_sz = mtk_api(_library).get_ffn_state_size(_runtime, layer);
        if (att_sz == 0 || wkv_sz == 0 || ffn_sz == 0) {
            return RWKV_ERROR_BACKEND | RWKV_ERROR_RUNTIME;
        }

        auto& att = (*states_ptr)[(size_t)layer * 3 + 0];
        auto& wkv = (*states_ptr)[(size_t)layer * 3 + 1];
        auto& ffn = (*states_ptr)[(size_t)layer * 3 + 2];

        att.resize(att_sz);
        wkv.resize(wkv_sz);
        ffn.resize(ffn_sz);

        if (!mtk_api(_library).get_att_state(_runtime, layer, att.data(), att.size())) return RWKV_ERROR_BACKEND | RWKV_ERROR_RUNTIME;
        if (!mtk_api(_library).get_wkv_state(_runtime, layer, wkv.data(), wkv.size())) return RWKV_ERROR_BACKEND | RWKV_ERROR_RUNTIME;
        if (!mtk_api(_library).get_ffn_state(_runtime, layer, ffn.data(), ffn.size())) return RWKV_ERROR_BACKEND | RWKV_ERROR_RUNTIME;
    }

    state = states_ptr;
    return RWKV_SUCCESS;
}

int mtk_np9_backend::set_state(std::any state) {
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

        if (!mtk_api(_library).set_att_state(_runtime, layer, att.data(), att.size())) return RWKV_ERROR_BACKEND | RWKV_ERROR_RUNTIME;
        if (!mtk_api(_library).set_wkv_state(_runtime, layer, wkv.data(), wkv.size())) return RWKV_ERROR_BACKEND | RWKV_ERROR_RUNTIME;
        if (!mtk_api(_library).set_ffn_state(_runtime, layer, ffn.data(), ffn.size())) return RWKV_ERROR_BACKEND | RWKV_ERROR_RUNTIME;
    }
    return RWKV_SUCCESS;
}

int mtk_np9_backend::free_state(std::any state) {
    state.reset();
    return RWKV_SUCCESS;
}

int mtk_np9_backend::zero_state() {
    if (_runtime == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    mtk_api(_library).reset(_runtime);
    return RWKV_SUCCESS;
}

int mtk_np9_backend::get_state_on_batch_slot(int slot, std::any &state) {
    if (_runtime == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    if (slot < 0) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto api = mtk_api(_library);
    if (slot == 0 && (api.get_att_state_slot == nullptr || api.get_wkv_state_slot == nullptr || api.get_ffn_state_slot == nullptr)) {
        return get_state(state);
    }
    if (api.get_att_state_slot == nullptr || api.get_wkv_state_slot == nullptr || api.get_ffn_state_slot == nullptr) {
        return RWKV_ERROR_UNSUPPORTED;
    }

    auto states_ptr = std::make_shared<std::vector<std::vector<uint8_t>>>();
    states_ptr->resize((size_t)n_layers * 3);
    for (int layer = 0; layer < n_layers; ++layer) {
        const size_t att_sz = api.get_att_state_size(_runtime, layer);
        const size_t wkv_sz = api.get_wkv_state_size(_runtime, layer);
        const size_t ffn_sz = api.get_ffn_state_size(_runtime, layer);
        if (att_sz == 0 || wkv_sz == 0 || ffn_sz == 0) {
            return RWKV_ERROR_BACKEND | RWKV_ERROR_RUNTIME;
        }

        auto& att = (*states_ptr)[(size_t)layer * 3 + 0];
        auto& wkv = (*states_ptr)[(size_t)layer * 3 + 1];
        auto& ffn = (*states_ptr)[(size_t)layer * 3 + 2];
        att.resize(att_sz);
        wkv.resize(wkv_sz);
        ffn.resize(ffn_sz);
        if (!api.get_att_state_slot(_runtime, layer, slot, att.data(), att.size())) return RWKV_ERROR_BACKEND | RWKV_ERROR_RUNTIME;
        if (!api.get_wkv_state_slot(_runtime, layer, slot, wkv.data(), wkv.size())) return RWKV_ERROR_BACKEND | RWKV_ERROR_RUNTIME;
        if (!api.get_ffn_state_slot(_runtime, layer, slot, ffn.data(), ffn.size())) return RWKV_ERROR_BACKEND | RWKV_ERROR_RUNTIME;
    }
    state = states_ptr;
    return RWKV_SUCCESS;
}

int mtk_np9_backend::set_state_on_batch_slot(int slot, std::any state) {
    if (_runtime == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    if (slot < 0) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto api = mtk_api(_library);
    if (slot == 0 && (api.set_att_state_slot == nullptr || api.set_wkv_state_slot == nullptr || api.set_ffn_state_slot == nullptr)) {
        return set_state(state);
    }
    if (api.set_att_state_slot == nullptr || api.set_wkv_state_slot == nullptr || api.set_ffn_state_slot == nullptr) {
        return RWKV_ERROR_UNSUPPORTED;
    }
    if (!state.has_value()) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }

    std::shared_ptr<std::vector<std::vector<uint8_t>>> states_ptr;
    try {
        states_ptr = std::any_cast<std::shared_ptr<std::vector<std::vector<uint8_t>>>>(state);
    } catch (const std::bad_any_cast&) {
        try {
            auto by_value = std::any_cast<std::vector<std::vector<uint8_t>>>(state);
            states_ptr = std::make_shared<std::vector<std::vector<uint8_t>>>(std::move(by_value));
        } catch (const std::bad_any_cast&) {
            return RWKV_ERROR_INVALID_PARAMETERS;
        }
    }
    if (!states_ptr || (int)states_ptr->size() != 3 * n_layers) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }
    for (int layer = 0; layer < n_layers; ++layer) {
        const auto& att = (*states_ptr)[(size_t)layer * 3 + 0];
        const auto& wkv = (*states_ptr)[(size_t)layer * 3 + 1];
        const auto& ffn = (*states_ptr)[(size_t)layer * 3 + 2];
        if (!api.set_att_state_slot(_runtime, layer, slot, att.data(), att.size())) return RWKV_ERROR_BACKEND | RWKV_ERROR_RUNTIME;
        if (!api.set_wkv_state_slot(_runtime, layer, slot, wkv.data(), wkv.size())) return RWKV_ERROR_BACKEND | RWKV_ERROR_RUNTIME;
        if (!api.set_ffn_state_slot(_runtime, layer, slot, ffn.data(), ffn.size())) return RWKV_ERROR_BACKEND | RWKV_ERROR_RUNTIME;
    }
    return RWKV_SUCCESS;
}

int mtk_np9_backend::zero_state_on_batch_slot(int slot) {
    if (_runtime == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto api = mtk_api(_library);
    if (api.zero_state_slot == nullptr) {
        return slot == 0 ? zero_state() : RWKV_ERROR_UNSUPPORTED;
    }
    if (!api.zero_state_slot(_runtime, slot)) {
        return RWKV_ERROR_BACKEND | RWKV_ERROR_RUNTIME;
    }
    return RWKV_SUCCESS;
}

int mtk_np9_backend::load_raw_states(std::vector<std::vector<half_float::half>> states) {
    // Used by rwkv-mobile rmpack state loader (one file per layer).
    // Interpret it as WKV state per layer, and zero ATTN/FFN states.
    if (_runtime == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    if ((int)states.size() != n_layers) {
        return RWKV_ERROR_INVALID_PARAMETERS;
    }

    const size_t att_sz = mtk_api(_library).get_att_state_size(_runtime, 0);
    const size_t wkv_sz = mtk_api(_library).get_wkv_state_size(_runtime, 0);
    const size_t ffn_sz = mtk_api(_library).get_ffn_state_size(_runtime, 0);
    if (att_sz == 0 || wkv_sz == 0 || ffn_sz == 0) {
        return RWKV_ERROR_BACKEND | RWKV_ERROR_RUNTIME;
    }

    for (int layer = 0; layer < n_layers; ++layer) {
        // Zero att/ffn.
        std::vector<uint8_t> zeros(att_sz, 0);
        if (!mtk_api(_library).set_att_state(_runtime, layer, zeros.data(), zeros.size())) return RWKV_ERROR_BACKEND | RWKV_ERROR_RUNTIME;
        zeros.assign(ffn_sz, 0);
        if (!mtk_api(_library).set_ffn_state(_runtime, layer, zeros.data(), zeros.size())) return RWKV_ERROR_BACKEND | RWKV_ERROR_RUNTIME;

        // Load wkv.
        const auto& wkv_half = states[layer];
        const size_t bytes = wkv_half.size() * sizeof(half_float::half);
        if (bytes != wkv_sz) {
            LOGE("[mtk_np9] load_raw_states: layer %d size mismatch: got=%zu, want=%zu\n", layer, bytes, wkv_sz);
            return RWKV_ERROR_INVALID_PARAMETERS;
        }
        if (!mtk_api(_library).set_wkv_state(_runtime, layer, wkv_half.data(), bytes)) return RWKV_ERROR_BACKEND | RWKV_ERROR_RUNTIME;
    }

    return RWKV_SUCCESS;
}

int mtk_np9_backend::release_model() {
    if (_runtime && _library.is_loaded()) {
        mtk_api(_library).release(_runtime);
        _runtime = nullptr;
    }
    _prefill_seq_len = 0;
    return RWKV_SUCCESS;
}

int mtk_np9_backend::release() {
    release_model();
    _library.close();
    return RWKV_SUCCESS;
}

} // namespace rwkvmobile
