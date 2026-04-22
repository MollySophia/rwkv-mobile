#include <fstream>
#include <filesystem>
#include <thread>
#include <algorithm>
#include <cstring>

#include "backend.h"
#include "llama_cpp_backend.h"
#include "c_api.h"
#include "llama.h"
#include "llama-model.h"
#include "llama-memory-recurrent.h"
#include "commondef.h"
#include "logger.h"

namespace rwkvmobile {

static constexpr uint32_t kReplayableStateMagic = 0x5350524c; // "LRPS"
static constexpr uint32_t kReplayableStateVersion = 1;

static int make_logits_tensor_view(float * logits_out, size_t count, Tensor1D &logits) {
    if (!logits_out) {
        return RWKV_ERROR_EVAL;
    }
    logits = Tensor1D::make(logits_out, TensorDType::F32, count);
    return RWKV_SUCCESS;
}

void llama_cpp_backend::initialize_supported_batch_sizes() {
#if defined(__ANDROID__)
    supported_batch_sizes = {1};
#else
    supported_batch_sizes.clear();
    supported_batch_sizes.reserve(kMaxBatchSlots);
    for (int i = 1; i <= kMaxBatchSlots; ++i) {
        supported_batch_sizes.push_back(i);
    }
#endif
}

int llama_cpp_backend::init(void * extra) {
    llama_log_set([](enum ggml_log_level level, const char * text, void * /* user_data */) {
        std::string log_msg = std::string(text);
        if (log_msg.empty()) {
            return;
        }
        while (log_msg.size() > 0 && log_msg[log_msg.size() - 1] == '\n') {
            log_msg = log_msg.substr(0, log_msg.size() - 1);
        }
        switch (level) {
            case GGML_LOG_LEVEL_ERROR:
                LOGE("%s", log_msg.c_str());
                break;
            case GGML_LOG_LEVEL_WARN:
                LOGW("%s", log_msg.c_str());
                break;
            case GGML_LOG_LEVEL_INFO:
                LOGI("%s", log_msg.c_str());
                break;
            case GGML_LOG_LEVEL_DEBUG:
                LOGD("%s", log_msg.c_str());
                break;
            default:
                break;
        }
    }, nullptr);

    return RWKV_SUCCESS;
}

int llama_cpp_backend::load_model(std::string model_path, void * extra) {
    llama_model_params model_params = llama_model_default_params();

#if defined(__APPLE__) || defined(__MACH__) || defined(GGML_USE_VULKAN)
    model_params.n_gpu_layers = 99;
#else
    model_params.n_gpu_layers = 0;
#endif
    model_params.progress_callback = nullptr;

    llama_cpp_args *args = nullptr;
    if (extra) {
        args = reinterpret_cast<llama_cpp_args*>(extra);
    }
    if (args) {
        model_params.n_gpu_layers = args->n_gpu_layers;
    }

    LOGI("n_gpu_layers: %d", model_params.n_gpu_layers);
    model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
    }

    llama_context_params ctx_params = llama_context_default_params();
    const uint32_t n_ctx_train = std::max<uint32_t>(1u, (uint32_t) llama_model_n_ctx_train(model));
    // RWKV mobile models can advertise very large metadata contexts; keep the
    // per-sequence working window aligned with the common 8k runtime target.
    const uint32_t n_ctx_per_seq = std::min<uint32_t>(n_ctx_train, 8192u);
    ctx_params.n_ctx = n_ctx_per_seq * (uint32_t) kMaxBatchSlots;
    ctx_params.n_seq_max = kMaxBatchSlots;
    ctx_params.kv_unified = false;
    ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
    }

// #ifdef __ANDROID__
//     // TODO: set according to the number of prime cores on the device
//     llama_set_n_threads(ctx, 2, 2);
// #endif

    vocab_size = model->vocab.n_tokens();
    hidden_size = llama_model_n_embd(model);
    num_heads = hidden_size / 64;
    n_layers = llama_model_n_layer(model);
    initialize_supported_batch_sizes();

    if (batch_decode_initialized) {
        llama_batch_free(batch_decode);
        batch_decode = {};
        batch_decode_initialized = false;
    }
    batch_decode = llama_batch_init(kMaxBatchSlots, 0, kMaxBatchSlots);
    batch_decode_initialized = true;
    pending_checkpoint_states.assign((size_t) kMaxBatchSlots, replayable_state{});
    return RWKV_SUCCESS;
}

int llama_cpp_backend::eval(int id, Tensor1D & logits) {
    llama_batch batch = llama_batch_get_one(&id, 1);
    if (llama_decode(ctx, batch) != 0) {
        return RWKV_ERROR_EVAL;
    }

    float * logits_out = llama_get_logits_ith(ctx, -1);
    int ret = make_logits_tensor_view(logits_out, (size_t) vocab_size, logits);
    if (ret != RWKV_SUCCESS) {
        return ret;
    }

    if (!pending_checkpoint_states.empty()) {
        pending_checkpoint_states[0] = replayable_state{};
    }

    return RWKV_SUCCESS;
}

int llama_cpp_backend::eval(std::vector<int> ids, Tensor1D & logits) {
    if (ids.empty()) {
        return RWKV_ERROR_BACKEND | RWKV_ERROR_INVALID_PARAMETERS;
    }

    if (ids.size() > 1) {
        std::vector<int> prefix(ids.begin(), ids.end() - 1);
        llama_batch prefix_batch = llama_batch_get_one(prefix.data(), prefix.size());
        if (llama_decode(ctx, prefix_batch) != 0) {
            return RWKV_ERROR_EVAL;
        }
    }

    return eval(ids.back(), logits);
}

int llama_cpp_backend::eval_batch(std::vector<std::vector<int>> ids, Tensor1D & logits) {
    std::vector<int> flat_ids;
    flat_ids.reserve(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i].size() != 1) {
            LOGE("llama_cpp_backend::eval_batch only supports single-token decode per slot");
            return RWKV_ERROR_UNSUPPORTED;
        }
        flat_ids.push_back(ids[i][0]);
    }
    return eval_batch_tokens(flat_ids, logits);
}

int llama_cpp_backend::eval_batch_tokens(const std::vector<int> &ids, Tensor1D & logits) {
    if (!ctx || !batch_decode_initialized) {
        return RWKV_ERROR_EVAL;
    }

    const int batch_size = (int) ids.size();
    if (batch_size <= 0 || batch_size > kMaxBatchSlots) {
        return RWKV_ERROR_UNSUPPORTED;
    }

    llama_memory_t mem = llama_get_memory(ctx);
    if (!mem) {
        return RWKV_ERROR_EVAL;
    }

    batch_decode.n_tokens = 0;
    for (int i = 0; i < batch_size; ++i) {
        const llama_seq_id seq_id = (llama_seq_id) i;
        llama_pos pos = llama_memory_seq_pos_max(mem, seq_id) + 1;
        if (pos < 0) {
            pos = 0;
        }

        batch_decode.token[i] = ids[(size_t) i];
        batch_decode.pos[i] = pos;
        batch_decode.n_seq_id[i] = 1;
        batch_decode.seq_id[i][0] = seq_id;
        batch_decode.logits[i] = 1;
        batch_decode.n_tokens++;
    }

    if (llama_decode(ctx, batch_decode) != 0) {
        return RWKV_ERROR_EVAL;
    }

    for (int i = 0; i < batch_size; ++i) {
        pending_checkpoint_states[(size_t) i] = replayable_state{};
    }

    float * logits_out = llama_get_logits(ctx);
    return make_logits_tensor_view(logits_out, (size_t) batch_size * (size_t) vocab_size, logits);
}

int llama_cpp_backend::eval_with_embeddings(const float *embeddings, int n_tokens, Tensor1D & logits) {
    int n_embd = llama_model_n_embd(model);

    llama_batch batch = {
        /*n_tokens       =*/ n_tokens,
        /*tokens         =*/ nullptr,
        /*embd           =*/ (float *)embeddings,
        /*pos            =*/ nullptr,
        /*n_seq_id       =*/ nullptr,
        /*seq_id         =*/ nullptr,
        /*logits         =*/ nullptr,
    };
    if (llama_decode(ctx, batch) != 0) {
        return RWKV_ERROR_EVAL;
    }
    float * logits_out = llama_get_logits_ith(ctx, -1);
    int ret = make_logits_tensor_view(logits_out, (size_t) vocab_size, logits);
    if (ret != RWKV_SUCCESS) {
        return ret;
    }
    if (!pending_checkpoint_states.empty()) {
        pending_checkpoint_states[0] = replayable_state{};
    }
    return RWKV_SUCCESS;
}

bool llama_cpp_backend::is_available() {
    return true;
}

int llama_cpp_backend::zero_state() {
    llama_memory_clear(llama_get_memory(ctx), true);
    pending_checkpoint_states.assign((size_t) kMaxBatchSlots, replayable_state{});
    return RWKV_SUCCESS;
}

int llama_cpp_backend::get_state(std::any &state) {
    return get_state_on_batch_slot(0, state);
}

int llama_cpp_backend::set_state(std::any state) {
    return set_state_on_batch_slot(0, state);
}

int llama_cpp_backend::free_state(std::any state) {
    if (!state.has_value()) {
        return RWKV_SUCCESS;
    }
    try {
        replayable_state & state_data = std::any_cast<replayable_state &>(state);
        state_data.pre_last_token_state.clear();
        state_data.seq_state.clear();
        state.reset();
    } catch (const std::bad_any_cast &) {
        try {
            std::vector<uint8_t> & state_mem = std::any_cast<std::vector<uint8_t> &>(state);
            state_mem.clear();
            state.reset();
        } catch (const std::bad_any_cast &) {
            return RWKV_ERROR_BACKEND | RWKV_ERROR_INVALID_PARAMETERS;
        }
    }
    return RWKV_SUCCESS;
}

int llama_cpp_backend::restore_replayable_state_on_slot(int slot, const replayable_state &state_data) {
    int ret = set_state_bytes_for_slot(slot, state_data.pre_last_token_state);
    if (ret != RWKV_SUCCESS) {
        return ret;
    }

    if (!state_data.has_last_token) {
        return RWKV_SUCCESS;
    }

    llama_memory_t mem = llama_get_memory(ctx);
    if (!mem) {
        return RWKV_ERROR_EVAL;
    }

    llama_token token = (llama_token) state_data.last_token;
    llama_pos pos = llama_memory_seq_pos_max(mem, (llama_seq_id) slot) + 1;
    if (pos < 0) {
        pos = 0;
    }
    int32_t n_seq_id = 1;
    llama_seq_id seq_id_storage[1] = { (llama_seq_id) slot };
    llama_seq_id * seq_id_ptrs[1] = { seq_id_storage };
    int8_t logits_flag = 1;
    llama_batch replay_batch = {
        /*n_tokens       =*/ 1,
        /*tokens         =*/ &token,
        /*embd           =*/ nullptr,
        /*pos            =*/ &pos,
        /*n_seq_id       =*/ &n_seq_id,
        /*seq_id         =*/ seq_id_ptrs,
        /*logits         =*/ &logits_flag,
    };

    if (llama_decode(ctx, replay_batch) != 0) {
        return RWKV_ERROR_EVAL;
    }

    return RWKV_SUCCESS;
}

int llama_cpp_backend::parse_runtime_state(std::any state, replayable_state &state_data) {
    state_data = replayable_state{};
    if (!state.has_value()) {
        return RWKV_SUCCESS;
    }

    try {
        state_data = std::any_cast<const replayable_state &>(state);
        return RWKV_SUCCESS;
    } catch (const std::bad_any_cast &) {
    }

    try {
        state_data.seq_state = std::any_cast<const std::vector<uint8_t> &>(state);
        return RWKV_SUCCESS;
    } catch (const std::bad_any_cast &) {
    }

    return RWKV_ERROR_BACKEND | RWKV_ERROR_INVALID_PARAMETERS;
}

int llama_cpp_backend::get_state_bytes_for_slot(int slot, std::vector<uint8_t> &state_bytes) {
    if (!ctx || slot < 0 || slot >= kMaxBatchSlots) {
        return RWKV_ERROR_BACKEND | RWKV_ERROR_INVALID_PARAMETERS;
    }

    const size_t state_size = llama_state_seq_get_size(ctx, (llama_seq_id) slot);
    state_bytes.resize(state_size);
    if (state_size == 0) {
        return RWKV_SUCCESS;
    }

    const size_t copied = llama_state_seq_get_data(ctx, state_bytes.data(), state_bytes.size(), (llama_seq_id) slot);
    if (copied != state_size) {
        LOGE("llama_cpp_backend::get_state_bytes_for_slot: copied %zu bytes, expected %zu", copied, state_size);
        state_bytes.clear();
        return RWKV_ERROR_EVAL;
    }

    return RWKV_SUCCESS;
}

int llama_cpp_backend::set_state_bytes_for_slot(int slot, const std::vector<uint8_t> &state_bytes) {
    if (!ctx || slot < 0 || slot >= kMaxBatchSlots) {
        return RWKV_ERROR_BACKEND | RWKV_ERROR_INVALID_PARAMETERS;
    }

    int ret = zero_state_on_batch_slot(slot);
    if (ret != RWKV_SUCCESS) {
        return ret;
    }

    if (state_bytes.empty()) {
        return RWKV_SUCCESS;
    }

    const size_t restored = llama_state_seq_set_data(ctx, state_bytes.data(), state_bytes.size(), (llama_seq_id) slot);
    if (restored != state_bytes.size()) {
        LOGE("llama_cpp_backend::set_state_bytes_for_slot: restored %zu bytes, expected %zu", restored, state_bytes.size());
        return RWKV_ERROR_EVAL;
    }

    return RWKV_SUCCESS;
}

int llama_cpp_backend::get_state_on_batch_slot(int slot, std::any &state) {
    replayable_state state_data;
    if ((size_t) slot < pending_checkpoint_states.size()) {
        state_data = pending_checkpoint_states[(size_t) slot];
    }
    if (!state_data.has_last_token && state_data.pre_last_token_state.empty()) {
        int ret = get_state_bytes_for_slot(slot, state_data.pre_last_token_state);
        if (ret != RWKV_SUCCESS) {
            return ret;
        }
    }
    state = std::move(state_data);
    return RWKV_SUCCESS;
}

int llama_cpp_backend::set_state_on_batch_slot(int slot, std::any state) {
    if (!state.has_value()) {
        return zero_state_on_batch_slot(slot);
    }
    replayable_state state_data;
    int ret = parse_runtime_state(state, state_data);
    if (ret != RWKV_SUCCESS) {
        return ret;
    }
    if (!state_data.seq_state.empty()) {
        ret = set_state_bytes_for_slot(slot, state_data.seq_state);
    } else {
        ret = restore_replayable_state_on_slot(slot, state_data);
    }
    if (ret != RWKV_SUCCESS) {
        return ret;
    }
    pending_checkpoint_states[(size_t) slot] = std::move(state_data);
    return RWKV_SUCCESS;
}

int llama_cpp_backend::zero_state_on_batch_slot(int slot) {
    if (!ctx || slot < 0 || slot >= kMaxBatchSlots) {
        return RWKV_ERROR_BACKEND | RWKV_ERROR_INVALID_PARAMETERS;
    }

    llama_memory_t mem = llama_get_memory(ctx);
    if (!mem) {
        return RWKV_ERROR_EVAL;
    }

    if (!llama_memory_seq_rm(mem, (llama_seq_id) slot, -1, -1)) {
        LOGW("llama_cpp_backend::zero_state_on_batch_slot: failed to fully clear seq %d", slot);
    }
    if ((size_t) slot < pending_checkpoint_states.size()) {
        pending_checkpoint_states[(size_t) slot] = replayable_state{};
    }
    return RWKV_SUCCESS;
}

int llama_cpp_backend::copy_state_between_batch_slots(int src_slot, int dst_slot) {
    if (!ctx || src_slot < 0 || src_slot >= kMaxBatchSlots || dst_slot < 0 || dst_slot >= kMaxBatchSlots) {
        return RWKV_ERROR_BACKEND | RWKV_ERROR_INVALID_PARAMETERS;
    }

    llama_memory_t mem = llama_get_memory(ctx);
    if (!mem) {
        return RWKV_ERROR_EVAL;
    }

    int ret = zero_state_on_batch_slot(dst_slot);
    if (ret != RWKV_SUCCESS) {
        return ret;
    }

    llama_memory_seq_cp(mem, (llama_seq_id) src_slot, (llama_seq_id) dst_slot, -1, -1);
    pending_checkpoint_states[(size_t) dst_slot] = pending_checkpoint_states[(size_t) src_slot];
    return RWKV_SUCCESS;
}

int llama_cpp_backend::load_raw_states(std::vector<std::vector<half_float::half>> states) {
    zero_state();
    Tensor1D logits;
    eval(0, logits);
    llama_memory_recurrent * mem = (llama_memory_recurrent *)llama_get_memory(ctx);
    for (int i = 0; i < n_layers; i++) {
        ggml_tensor * r = mem->r_l[i];
        ggml_tensor * s = mem->s_l[i];
        const int slot_count = std::max(1u, llama_n_seq_max(ctx));
        const int s_state_elems = (int)(s->ne[0] / slot_count);
        const int r_state_elems = (int)(r->ne[0] / slot_count);

        if (s_state_elems != hidden_size * (hidden_size / num_heads)) {
            LOGE("state size mismatch, expected %d, got %d", hidden_size * (hidden_size / num_heads), s_state_elems);
            return RWKV_ERROR_BACKEND | RWKV_ERROR_INVALID_PARAMETERS;
        }

        std::vector<float> state_f32((size_t) s_state_elems);
        for (int j = 0; j < s_state_elems; j++) {
            state_f32[j] = states[i][j];
        }
        ggml_backend_tensor_set(s, state_f32.data(), 0, state_f32.size() * sizeof(float));
        ggml_backend_tensor_memset(r, 0, 0, (size_t) r_state_elems * sizeof(float));
    }

    return RWKV_SUCCESS;
}

int llama_cpp_backend::serialize_runtime_state(std::any state, std::vector<uint8_t> &data) {
    replayable_state state_data;
    int ret = parse_runtime_state(state, state_data);
    if (ret != RWKV_SUCCESS) {
        return ret;
    }

    data.clear();
    const uint64_t state_size = (uint64_t) state_data.pre_last_token_state.size();
    const uint64_t seq_state_size = (uint64_t) state_data.seq_state.size();
    const uint8_t has_last_token = state_data.has_last_token ? 1 : 0;

    auto append_bytes = [&](const void * ptr, size_t size) {
        const auto * begin = (const uint8_t *) ptr;
        data.insert(data.end(), begin, begin + size);
    };

    append_bytes(&kReplayableStateMagic, sizeof(kReplayableStateMagic));
    append_bytes(&kReplayableStateVersion, sizeof(kReplayableStateVersion));
    append_bytes(&has_last_token, sizeof(has_last_token));
    append_bytes(&state_data.last_token, sizeof(state_data.last_token));
    append_bytes(&state_size, sizeof(state_size));
    append_bytes(&seq_state_size, sizeof(seq_state_size));
    if (state_size > 0) {
        append_bytes(state_data.pre_last_token_state.data(), (size_t) state_size);
    }
    if (seq_state_size > 0) {
        append_bytes(state_data.seq_state.data(), (size_t) seq_state_size);
    }
    return RWKV_SUCCESS;
}

int llama_cpp_backend::deserialize_runtime_state(std::vector<uint8_t> &data, std::any &state) {
    if (data.size() >= sizeof(uint32_t) * 2 + sizeof(uint8_t) + sizeof(int32_t) + sizeof(uint64_t) * 2) {
        size_t offset = 0;
        auto read_bytes = [&](void * ptr, size_t size) {
            memcpy(ptr, data.data() + offset, size);
            offset += size;
        };

        uint32_t magic = 0;
        uint32_t version = 0;
        uint8_t has_last_token = 0;
        int32_t last_token = 0;
        uint64_t state_size = 0;
        uint64_t seq_state_size = 0;
        read_bytes(&magic, sizeof(magic));
        read_bytes(&version, sizeof(version));
        read_bytes(&has_last_token, sizeof(has_last_token));
        read_bytes(&last_token, sizeof(last_token));
        read_bytes(&state_size, sizeof(state_size));
        read_bytes(&seq_state_size, sizeof(seq_state_size));

        if (magic == kReplayableStateMagic && version == kReplayableStateVersion &&
            offset + state_size + seq_state_size <= data.size()) {
            replayable_state state_data;
            state_data.has_last_token = has_last_token != 0;
            state_data.last_token = (int) last_token;
            state_data.pre_last_token_state.resize((size_t) state_size);
            if (state_size > 0) {
                memcpy(state_data.pre_last_token_state.data(), data.data() + offset, (size_t) state_size);
                offset += (size_t) state_size;
            }
            state_data.seq_state.resize((size_t) seq_state_size);
            if (seq_state_size > 0) {
                memcpy(state_data.seq_state.data(), data.data() + offset, (size_t) seq_state_size);
            }
            state = std::any(std::move(state_data));
            return RWKV_SUCCESS;
        }
    }

    state = std::any(std::vector<uint8_t>(data));
    return RWKV_SUCCESS;
}

int llama_cpp_backend::release_model() {
    if (batch_decode_initialized) {
        llama_batch_free(batch_decode);
        batch_decode = {};
        batch_decode_initialized = false;
    }
    if (ctx) {
        llama_free(ctx);
        ctx = nullptr;
    }
    if (model) {
        llama_model_free(model);
        model = nullptr;
    }
    pending_checkpoint_states.clear();
    return RWKV_SUCCESS;
}

int llama_cpp_backend::release() {
    return RWKV_SUCCESS;
}

} // namespace rwkvmobile
