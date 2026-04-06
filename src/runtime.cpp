#include "runtime.h"
#include "backend.h"
#include "logger.h"
#include "utils.h"
#include <functional>
#include <filesystem>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <thread>
#include "rmpack.h"
#include "utils.h"
#ifdef ENABLE_WEBRWKV
#include "web_rwkv_backend.h"
#endif

#ifdef ENABLE_NCNN
#include "ncnn_rwkv_backend.h"
#endif

#ifdef ENABLE_LLAMACPP
#include "llama_cpp_backend.h"
#endif

#ifdef ENABLE_QNN
#include "qnn_backend.h"
#endif

#ifdef ENABLE_MNN
#include "mnn_rwkv_backend.h"
#endif

#ifdef ENABLE_MTK_NP7
#include "mtk_np7_backend.h"
#endif

#ifdef ENABLE_COREML
#include "coreml_rwkv_backend.h"
#endif

#ifdef ENABLE_MLX
#include "mlx_rwkv_backend.h"
#endif

#if defined(ENABLE_VISION)
#include "multimodal/vision/vision_encoder.h"
#endif

#if defined(ENABLE_WHISPER)
#include "multimodal/whisper/whisper_encoder.h"
#endif

#if defined(ENABLE_TTS)
#include "frontend_utils.h"
#include "tts_properties.h"
#endif

#if defined(ENABLE_TTS) || defined(ENABLE_WHISPER)
#include "audio.h"
#endif

namespace rwkvmobile {

const int chinese_tokens_start = 10250;
const int chinese_tokens_end = 18493;

inline void mask_thinking_tag(Tensor1D &logits) {
    tensor1d_set_f32(logits, 11, -1e9f); // '\n'
    tensor1d_set_f32(logits, 61, -1e9f); // '<'
    tensor1d_set_f32(logits, 261, -1e9f); // '\n\n'
    tensor1d_set_f32(logits, 295, -1e9f); // ' <'
    tensor1d_set_f32(logits, 0, -1e9f); // <EOD>
}

inline void mask_non_chinese_tokens(Tensor1D &logits, int num_vocab) {
    int current_token = 0, current_range_idx = 0;
    while (current_token < num_vocab) {
        if (current_token >= chinese_tokens_start && current_token <= chinese_tokens_end) {
            current_token = chinese_tokens_end + 1;
        }
        tensor1d_set_f32(logits, (size_t)current_token, -1e9f);
        current_token++;
    }
}

void Runtime::_record_speed_sample(ModelInstance& model, bool is_prefill, int tokens, int64_t duration_us) {
    if (tokens <= 0 || duration_us <= 0) {
        return;
    }
    ModelInstance::SpeedSample sample;
    sample.tokens = tokens;
    sample.duration_us = duration_us;

    std::lock_guard<std::mutex> lock(model.speed_samples_mutex);
    auto& q = is_prefill ? model.prefill_samples_us : model.decode_samples_us;
    q.push_back(sample);
    while (q.size() > _speed_samples_max) {
        q.pop_front();
    }
}

void Runtime::_clear_speed_samples(ModelInstance& model) {
    std::lock_guard<std::mutex> lock(model.speed_samples_mutex);
    model.decode_samples_us.clear();
    model.prefill_samples_us.clear();
}

double Runtime::_compute_trimmed_mean_speed_tokens_per_s(
    const std::deque<ModelInstance::SpeedSample>& samples,
    double trim_ratio_total
) {
    if (samples.empty()) {
        return 0.0;
    }

    std::vector<double> speeds;
    speeds.reserve(samples.size());
    for (const auto& s : samples) {
        if (s.tokens <= 0 || s.duration_us <= 0) {
            continue;
        }
        speeds.push_back((double)s.tokens * 1e6 / (double)s.duration_us);
    }
    if (speeds.empty()) {
        return 0.0;
    }

    std::sort(speeds.begin(), speeds.end());
    const size_t n = speeds.size();

    // Keep middle (1 - trim_ratio_total). Default is 90% (trim 5% on each side).
    const double half_trim = std::max(0.0, std::min(0.5, trim_ratio_total * 0.5));
    size_t trim_each_side = (size_t)std::floor((double)n * half_trim);
    if (trim_each_side * 2 >= n) {
        trim_each_side = 0;
    }

    const size_t begin = trim_each_side;
    const size_t end = n - trim_each_side;
    double sum = 0.0;
    for (size_t i = begin; i < end; i++) {
        sum += speeds[i];
    }
    return sum / (double)(end - begin);
}

std::string backend_enum_to_str(int backend) {
    switch (backend) {
        case RWKV_BACKEND_WEBRWKV:
            return "web-rwkv";
        case RWKV_BACKEND_NCNN:
            return "ncnn";
        case RWKV_BACKEND_LLAMACPP:
            return "llama.cpp";
        case RWKV_BACKEND_QNN:
            return "qnn";
        case RWKV_BACKEND_MNN:
            return "mnn";
        case RWKV_BACKEND_MTK_NP7:
            return "mtk_np7";
        case RWKV_BACKEND_COREML:
            return "coreml";
        case RWKV_BACKEND_MLX:
            return "mlx";
        default:
            return "unknown";
    }
}

int backend_str_to_enum(std::string backend) {
    if (backend == "web-rwkv") {
        return RWKV_BACKEND_WEBRWKV;
    } else if (backend == "ncnn") {
        return RWKV_BACKEND_NCNN;
    } else if (backend == "llama.cpp") {
        return RWKV_BACKEND_LLAMACPP;
    } else if (backend == "qnn") {
        return RWKV_BACKEND_QNN;
    } else if (backend == "mnn") {
        return RWKV_BACKEND_MNN;
    } else if (backend == "mtk_np7") {
        return RWKV_BACKEND_MTK_NP7;
    } else if (backend == "coreml") {
        return RWKV_BACKEND_COREML;
    } else if (backend == "mlx") {
        return RWKV_BACKEND_MLX;
    }
    return -1;
}

void Runtime::_prefill_progress_start(int model_id, int total_tokens) {
    if (_models.find(model_id) == _models.end()) {
        return;
    }

    auto &model = _models.at(model_id);
    double estimated_speed = model->backend ? model->backend->get_prefill_speed() : -1.0;
    if (estimated_speed <= 0.0) {
        std::lock_guard<std::mutex> speed_lock(model->speed_samples_mutex);
        estimated_speed = _compute_trimmed_mean_speed_tokens_per_s(model->prefill_samples_us, _speed_trim_ratio_total);
    }
    if (estimated_speed <= 0.0 && _prefill_speed > 0.0) {
        estimated_speed = _prefill_speed;
    }
    if (estimated_speed <= 0.0) {
        // Conservative fallback so the UI still shows progress on first use.
        estimated_speed = 512.0;
    }

    std::lock_guard<std::mutex> progress_lock(model->prefill_progress_mutex);
    model->current_prefill_total_tokens = total_tokens;
    model->current_prefill_finished_tokens = 0;
    model->prefill_progress = total_tokens > 0 ? 0.01 : 0.0;
    model->prefill_progress_started_at = std::chrono::steady_clock::now();
    model->prefill_estimated_total_us = std::max<int64_t>(
        50 * 1000,
        (int64_t) ((double) total_tokens * 1000000.0 / estimated_speed)
    );
}

void Runtime::_prefill_progress_finish(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return;
    }

    auto &model = _models.at(model_id);
    std::lock_guard<std::mutex> progress_lock(model->prefill_progress_mutex);
    model->current_prefill_finished_tokens = std::max(0, model->current_prefill_total_tokens);
    model->current_prefill_total_tokens = -1;
    model->prefill_estimated_total_us = 0;
    model->prefill_progress = 1.0;
}

int Runtime::_get_prefill_checkpoint_interval(int total_tokens) const {
    if (total_tokens <= 0) {
        return _prefill_chunk_size;
    }
    if (total_tokens <= 512) {
        return total_tokens;
    }
    if (total_tokens <= 2048) {
        return 512;
    }
    if (total_tokens <= 8192) {
        return 1024;
    }
    if (total_tokens <= 32768) {
        return 2048;
    }
    return 4096;
}

int Runtime::load_model(std::string model_path, std::string backend_name, std::string tokenizer_path, void * extra) {
    int ret_model_id = -1;
    int backend_id = backend_str_to_enum(backend_name);
    if (backend_id < 0) {
        LOGE("Invalid backend name: %s\n", backend_name.c_str());
        return ret_model_id;
    }

    auto model_instance = std::make_unique<ModelInstance>();
    if (model_instance == nullptr) {
        LOGE("Failed to allocate memory for model instance\n");
        return ret_model_id;
    }

    // 1. Create and initialize backend
    if (backend_id == RWKV_BACKEND_WEBRWKV) {
#ifdef ENABLE_WEBRWKV
        model_instance->backend = std::unique_ptr<execution_provider, std::function<void(execution_provider*)>>(new web_rwkv_backend,
            [](execution_provider *p) { delete (web_rwkv_backend*)p; });
#else
        LOGE("WebRWKV backend is not supported on this platform\n");
        return ret_model_id;
#endif
    } else if (backend_id == RWKV_BACKEND_NCNN) {
#ifdef ENABLE_NCNN
        model_instance->backend = std::unique_ptr<execution_provider, std::function<void(execution_provider*)>>(new ncnn_rwkv_backend,
            [](execution_provider *p) { delete (ncnn_rwkv_backend*)p; });
#else
        LOGE("NCNN backend is not supported on this platform\n");
        return ret_model_id;
#endif
    } else if (backend_id == RWKV_BACKEND_LLAMACPP) {
#ifdef ENABLE_LLAMACPP
        model_instance->backend = std::unique_ptr<execution_provider, std::function<void(execution_provider*)>>(new llama_cpp_backend,
            [](execution_provider *p) { delete (llama_cpp_backend*)p; });
#else
        LOGE("LLaMa.cpp backend is not supported on this platform\n");
        return ret_model_id;
#endif
    } else if (backend_id == RWKV_BACKEND_QNN) {
#ifdef ENABLE_QNN
        model_instance->backend = std::unique_ptr<execution_provider, std::function<void(execution_provider*)>>(new qnn_backend,
            [](execution_provider *p) { delete (qnn_backend*)p; });
#else
        LOGE("QNN backend is not supported on this platform\n");
        return ret_model_id;
#endif
    } else if (backend_id == RWKV_BACKEND_MNN) {
#ifdef ENABLE_MNN
        model_instance->backend = std::unique_ptr<execution_provider, std::function<void(execution_provider*)>>(new mnn_rwkv_backend,
            [](execution_provider *p) { delete (mnn_rwkv_backend*)p; });
#else
        LOGE("MNN backend is not supported on this platform\n");
        return ret_model_id;
#endif
    } else if (backend_id == RWKV_BACKEND_MTK_NP7) {
#ifdef ENABLE_MTK_NP7
        model_instance->backend = std::unique_ptr<execution_provider, std::function<void(execution_provider*)>>(new mtk_np7_backend,
            [](execution_provider *p) { delete (mtk_np7_backend*)p; });
#else
        LOGE("mtk_np7 backend is not supported on this platform\n");
        return ret_model_id;
#endif
    } else if (backend_id == RWKV_BACKEND_COREML) {
#ifdef ENABLE_COREML
        model_instance->backend = std::unique_ptr<execution_provider, std::function<void(execution_provider*)>>(new coreml_rwkv_backend,
            [](execution_provider *p) { delete (coreml_rwkv_backend*)p; });
#else
        LOGE("CoreML backend is not supported on this platform\n");
        return ret_model_id;
#endif
    } else if (backend_id == RWKV_BACKEND_MLX) {
#ifdef ENABLE_MLX
        model_instance->backend = std::unique_ptr<execution_provider, std::function<void(execution_provider*)>>(new mlx_rwkv_backend,
            [](execution_provider *p) { delete (mlx_rwkv_backend*)p; });
#else
        LOGE("MLX backend is not supported on this platform\n");
        return ret_model_id;
#endif
    } else {
        LOGE("Unsupported backend: %s\n", backend_name.c_str());
        return ret_model_id;
    }

    int ret = model_instance->backend->init(extra);
    if (ret) {
        LOGE("Failed to initialize backend: %s, errno = %d\n", backend_name.c_str(), ret);
        return ret_model_id;
    }

    // 2. Load model (expose backend for progress polling during async load)
    {
        std::lock_guard<std::mutex> lock(_loading_backend_mutex);
        _loading_backend = model_instance->backend.get();
    }
    ret = model_instance->backend->load_model(model_path, extra);
    {
        std::lock_guard<std::mutex> lock(_loading_backend_mutex);
        _loading_backend = nullptr;
    }
    if (ret) {
        LOGE("Failed to load model from: %s, errno = %d\n", model_path.c_str(), ret);
        return -ret;
    }

    int next_model_id = 0;
    while (_models.find(next_model_id) != _models.end()) {
        next_model_id++;
    }

    LOGI("Loaded model from: %s as model_id = %d\n", model_path.c_str(), next_model_id);
    LOGI("Model num_layers: %d, num_heads: %d, hidden_size: %d, vocab_size: %d\n",
         model_instance->backend->n_layers, model_instance->backend->num_heads, model_instance->backend->hidden_size, model_instance->backend->vocab_size);
    model_instance->backend->zero_state();
    model_instance->backend->state_root = std::make_unique<state_node>();
    if (model_instance->backend->state_root == nullptr) {
        return ret_model_id;
    }

    model_instance->backend->state_root->activation_count = 10;
    model_instance->backend->get_state(model_instance->backend->state_root->state);
    model_instance->backend->state_root->ids = std::vector<int>();
    model_instance->backend->state_root->logits = std::vector<float>(model_instance->backend->vocab_size, 0);

    // 3. Load tokenizer
    if (!tokenizer_path.empty()) {
        model_instance->tokenizer = std::unique_ptr<tokenizer_base, std::function<void(tokenizer_base*)>>(new trie_tokenizer,
            [](tokenizer_base *p) { delete (trie_tokenizer*)p; });
        if (model_instance->tokenizer == nullptr) {
            return ret_model_id;
        }
        ret = model_instance->tokenizer->load(tokenizer_path);
        if (ret) {
            LOGE("[LOAD_MODEL] Failed to load tokenizer for model ID %d", next_model_id);
            return ret_model_id;
        }
    }

    // 4. Create sampler
    model_instance->sampler = std::make_unique<NucleusSampler>();
    if (model_instance->sampler == nullptr) {
        return ret_model_id;
    }

    // 5. Store the instance and return its ID
    ret_model_id = next_model_id;
    _models[ret_model_id] = std::move(model_instance);
    _models[ret_model_id]->model_path = model_path;
    _models[ret_model_id]->backend_name = backend_name;
    _models[ret_model_id]->tokenizer_path = tokenizer_path;

    // pre-prefill "User"
    prefill_to_cache(ret_model_id, _models[ret_model_id]->bos_token + _models[ret_model_id]->user_role);

    return ret_model_id;
}

int Runtime::release_model(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    _models.erase(model_id);
    return RWKV_SUCCESS;
}

void Runtime::start_load_model_async() {
    _load_model_in_progress.store(true);
    {
        std::lock_guard<std::mutex> lock(_load_progress_fallback_mutex);
        _load_progress_fallback = 0.f;
        _load_progress_fallback_step = 0.1f;
    }
}

void Runtime::set_load_model_result(int result_code, int model_id) {
    std::lock_guard<std::mutex> lock(_load_model_result_mutex);
    _load_model_result_code = result_code;
    _load_model_result_id = model_id;
    _load_model_in_progress.store(false);
}

void Runtime::get_load_model_result(int& result_code, int& model_id) const {
    std::lock_guard<std::mutex> lock(_load_model_result_mutex);
    result_code = _load_model_result_code;
    model_id = _load_model_result_id;
}

float Runtime::get_load_model_progress() const {
    execution_provider* ep = nullptr;
    {
        std::lock_guard<std::mutex> lock(_loading_backend_mutex);
        ep = _loading_backend;
    }
    if (ep) {
        float p = ep->get_load_progress();
        if (p >= 0.f) {
            return std::max(0.f, std::min(1.f, p));
        }
        if (_load_model_in_progress.load()) {
            std::lock_guard<std::mutex> lock(_load_progress_fallback_mutex);
            auto progress_old = _load_progress_fallback;
            _load_progress_fallback = std::min(0.999f, _load_progress_fallback + _load_progress_fallback_step);
            _load_progress_fallback_step = std::max(0.001f, _load_progress_fallback_step * 0.9f);
            return progress_old;
        }
    }
    if (_load_model_in_progress.load()) {
        std::lock_guard<std::mutex> lock(_load_progress_fallback_mutex);
        auto progress_old = _load_progress_fallback;
        _load_progress_fallback = std::min(0.999f, _load_progress_fallback + _load_progress_fallback_step);
        _load_progress_fallback_step = std::max(0.001f, _load_progress_fallback_step * 0.9f);
        return progress_old;
    }
    return 1.0f;
}

#ifdef ENABLE_VISION
int Runtime::load_vision_encoder(int model_id, std::string model_path, std::string adapter_path) {
    if (_models.find(model_id) == _models.end()) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);
    model->multimodal_encoder = std::make_unique<VisionEncoder>();
    if (model->multimodal_encoder == nullptr) {
        return RWKV_ERROR_ALLOC;
    }
    return model->multimodal_encoder->load_model(model_path, adapter_path);
}

int Runtime::release_vision_encoder(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);
    model->multimodal_encoder.reset();
    return RWKV_SUCCESS;
}
#endif

int Runtime::set_image_unique_identifier(std::string unique_identifier) {
    _image_unique_identifier = unique_identifier;
    return RWKV_SUCCESS;
}

#ifdef ENABLE_WHISPER
int Runtime::load_whisper_encoder(int model_id, std::string model_path) {
    if (_models.find(model_id) == _models.end()) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);
    model->multimodal_encoder = std::make_unique<WhisperEncoder>();
    if (model->multimodal_encoder == nullptr) {
        return RWKV_ERROR_ALLOC;
    }
    return model->multimodal_encoder->load_model(model_path, "");
}

int Runtime::release_whisper_encoder(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);
    model->multimodal_encoder.reset();
    return RWKV_SUCCESS;
}
#endif

int Runtime::get_available_backend_ids(std::vector<int> &backend_ids) {
    backend_ids = std::vector<int>();

#ifdef ENABLE_WEBRWKV
    // Snapdragon platform doesn't support WebRWKV backend yet
    if (_soc_detect.get_platform_type() != PLATFORM_SNAPDRAGON) {
        backend_ids.push_back(RWKV_BACKEND_WEBRWKV);
    }
#endif

#ifdef ENABLE_NCNN
    backend_ids.push_back(RWKV_BACKEND_NCNN);
#endif

#ifdef ENABLE_LLAMACPP
    backend_ids.push_back(RWKV_BACKEND_LLAMACPP);
#endif

#ifdef ENABLE_QNN
    if (_soc_detect.get_platform_type() == PLATFORM_SNAPDRAGON) {
        auto supported_soc_names = std::vector<std::string>{"SM8650", "SM8635", "SM8550", "SM8475"};
        if (std::find(supported_soc_names.begin(), supported_soc_names.end(), _soc_detect.get_soc_partname()) != supported_soc_names.end()) {
            backend_ids.push_back(RWKV_BACKEND_QNN);
        }
    }
#endif

#ifdef ENABLE_MTK_NP7
    if (_soc_detect.get_platform_type() == PLATFORM_MEDIATEK) {
        backend_ids.push_back(RWKV_BACKEND_MTK_NP7);
    }
#endif

#ifdef ENABLE_COREML
    backend_ids.push_back(RWKV_BACKEND_COREML);
#endif

#ifdef ENABLE_MLX
    backend_ids.push_back(RWKV_BACKEND_MLX);
#endif

    return RWKV_SUCCESS;
}

std::string Runtime::get_available_backends_str() {
    std::vector<int> backend_ids;
    get_available_backend_ids(backend_ids);
    std::string ret = "";
    for (auto id : backend_ids) {
        ret += backend_enum_to_str(id) + ",";
    }
    return ret;
}

int Runtime::load_initial_state(int model_id, std::string state_path) {
    if (_models.find(model_id) == _models.end()) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);
    if (state_path.find(".rmpack") == std::string::npos) {
        LOGE("the specified state file is not a rmpack file\n");
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    std::string initial_state_str = "<state src=\"" + state_path + "\">";
    std::vector<int> initial_state_ids = model->tokenizer->encode(initial_state_str);

    for (int i = 0; i < model->backend->state_root->children.size(); i++) {
        if (model->backend->state_root->children[i]->ids == initial_state_ids) {
            return RWKV_SUCCESS;
        }
    }

    std::any initial_state;
    RMPackReader state_pack(state_path);
    int hidden_size_config = state_pack.getConfig()["hidden_size"];
    auto files = state_pack.getFiles();
    if (files.size() != model->backend->n_layers) {
        LOGE("state file has %d layers, but model has %d layers\n", (int)files.size(), model->backend->n_layers);
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    if (hidden_size_config != model->backend->hidden_size) {
        LOGE("state file has hidden size %d, but model has hidden size %d\n", hidden_size_config, model->backend->hidden_size);
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    size_t state_size = state_pack.getFileSize(files[0].filename);

    std::vector<std::vector<half_float::half>> states(model->backend->n_layers);
    for (int i = 0; i < model->backend->n_layers; i++) {
        states[i].resize(state_size / sizeof(half_float::half));
        auto data = state_pack.readFileToMemory(files[i].filename);
        std::copy_n(reinterpret_cast<const uint8_t*>(data), state_size, reinterpret_cast<uint8_t*>(states[i].data()));
        state_pack.freeFileMemory(files[i].filename);
    }

    model->backend->load_raw_states(states);
    model->backend->get_state(initial_state);
    // make a new constant state node
    model->backend->state_root->children.push_back(std::make_unique<state_node>(initial_state, initial_state_ids, std::vector<float>(model->backend->vocab_size, 0), true));
    model->backend->state_root->children.back()->activation_count = 50;
    return RWKV_SUCCESS;
}

void Runtime::unload_initial_state(int model_id, std::string state_path) {
    if (_models.find(model_id) == _models.end()) {
        return;
    }
    auto &model = _models.at(model_id);
    std::string initial_state_str = "<state src=\"" + state_path + "\">";
    std::vector<int> initial_state_ids = model->tokenizer->encode(initial_state_str);
    for (int i = 0; i < model->backend->state_root->children.size(); i++) {
        if (model->backend->state_root->children[i]->ids == initial_state_ids) {
            model->backend->state_root->children.erase(model->backend->state_root->children.begin() + i);
            break;
        }
    }
}

int Runtime::eval_logits(int model_id, int id, Tensor1D & logits) {
    if (_models.find(model_id) == _models.end()) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);
    if (model->backend == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto start = std::chrono::high_resolution_clock::now();
    int ret = model->backend->eval(id, logits);
    auto end = std::chrono::high_resolution_clock::now();
    const int64_t duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    _record_speed_sample(*model, /*is_prefill=*/false, /*tokens=*/1, duration_us);
    return ret;
}

int Runtime::eval_logits(int model_id, std::vector<int> ids, Tensor1D & logits) {
    if (_models.find(model_id) == _models.end()) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);
    if (model->backend == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    auto start = std::chrono::high_resolution_clock::now();
    int i = 0;
    int ret;
    for (; i + _prefill_chunk_size <= ids.size(); i += _prefill_chunk_size) {
        auto ids_chunk = std::vector<int>(ids.begin() + i, ids.begin() + i + _prefill_chunk_size);
        ret = model->backend->eval(ids_chunk, logits);
        if (ret != RWKV_SUCCESS) return ret;
        if (model->current_prefill_total_tokens > 0) {
            std::lock_guard<std::mutex> progress_lock(model->prefill_progress_mutex);
            model->current_prefill_finished_tokens += _prefill_chunk_size;
            model->prefill_progress = (double)model->current_prefill_finished_tokens / model->current_prefill_total_tokens;
            LOGD("Update prefill_progress = %f", model->prefill_progress);
        }
    }
    if (i < ids.size()) {
        auto ids_left = std::vector<int>(ids.begin() + i, ids.end());
        ret = model->backend->eval(ids_left, logits);
        if (model->current_prefill_total_tokens > 0) {
            std::lock_guard<std::mutex> progress_lock(model->prefill_progress_mutex);
            model->current_prefill_finished_tokens += ids_left.size();
            model->prefill_progress = (double)model->current_prefill_finished_tokens / model->current_prefill_total_tokens;
            LOGD("Update prefill_progress = %f", model->prefill_progress);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    const int64_t duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    _record_speed_sample(*model, /*is_prefill=*/true, /*tokens=*/(int)ids.size(), duration_us);
    return ret;
}

int Runtime::eval_logits_with_embeddings(int model_id, const float *embeddings, int n_tokens, Tensor1D & logits) {
    if (_models.find(model_id) == _models.end()) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);
    if (model->backend == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto start = std::chrono::high_resolution_clock::now();
    auto ret = model->backend->eval_with_embeddings(embeddings, n_tokens, logits);
    auto end = std::chrono::high_resolution_clock::now();
    const int64_t duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    _record_speed_sample(*model, /*is_prefill=*/(n_tokens > 1), /*tokens=*/n_tokens, duration_us);
    return ret;
}

int Runtime::eval_logits_batch_decode(int model_id, std::vector<int> ids, Tensor1D & logits) {
    if (_models.find(model_id) == _models.end()) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);
    if (model->backend == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<int>> ids_batch(ids.size());
    for (int i = 0; i < ids.size(); i++) {
        ids_batch[i] = std::vector<int>(1);
        ids_batch[i][0] = ids[i];
    }

    int ret = model->backend->eval_batch(ids_batch, logits);
    auto end = std::chrono::high_resolution_clock::now();
    const int64_t duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    _record_speed_sample(*model, /*is_prefill=*/false, /*tokens=*/(int)ids.size(), duration_us);
    return ret;
}

std::vector<int> Runtime::get_supported_batch_sizes(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return {};
    }
    auto &model = _models.at(model_id);
    return model->backend->supported_batch_sizes;
}

std::string Runtime::apply_chat_template(int model_id, std::vector<std::string> inputs, bool enable_reasoning, bool add_generation_prompt, std::vector<std::string> roles_map) {
    if (_models.find(model_id) == _models.end()) {
        return "";
    }
    auto &model = _models.at(model_id);

    static auto replace_text = [](const std::string& text, const std::string& old_str, const std::string& new_str) -> std::string {
        std::string result = text;
        size_t pos = 0;
        while ((pos = result.find(old_str, pos)) != std::string::npos) {
            result.replace(pos, old_str.length(), new_str);
            pos += new_str.length();
        }
        return result;
    };

    auto space_after_roles = get_space_after_roles(model_id);
    auto normalize_role = [&](const std::string &role) -> std::string {
        if (role == "user") {
            return model->user_role;
        }
        if (role == "assistant") {
            return model->response_role;
        }
        if (role == "system") {
            return model->system_role;
        }
        return role;
    };
    std::vector<std::string> resolved_roles;
    if (roles_map.size() == inputs.size()) {
        resolved_roles = roles_map;
    } else {
        resolved_roles.resize(inputs.size());
        for (size_t i = 0; i < inputs.size(); i++) {
            resolved_roles[i] = (i % 2 == 0) ? "user" : "assistant";
        }
    }

    std::string text = model->prompt;
    for (size_t i = 0; i < inputs.size(); i++) {
        std::string content = inputs[i];

        std::string role = normalize_role(resolved_roles[i]);
        if (role == model->user_role || role == "User" || role == model->system_role || role == "System") {
            content = replace_text(content, "\r\n", "\n");
            content = replace_text(content, "\n\n", "\n");
        }

        text += model->bos_token + role + ":" + (space_after_roles ? " " : "") + content;
        if (i != inputs.size() - 1) {
            text += model->eos_token;
        }
    }

    if (!inputs.empty() && add_generation_prompt) {
        text += model->eos_token;
        std::string last_role = normalize_role(resolved_roles.back());
        if (last_role == model->user_role) {
            text += model->bos_token + model->response_role + ":";
            if (enable_reasoning) {
                text += (space_after_roles ? " " : "") + model->thinking_token;
            }
        } else if (last_role == model->response_role) {
            text += model->bos_token + model->user_role + ":";
        }
    }
    return text;
}

std::vector<Runtime::TokenChunk> Runtime::split_text_by_image_and_token_num(const std::string text, int max_tokens_per_chunk, int model_id) {
    std::vector<TokenChunk> chunks;
    std::string remaining_text = text;

    // Helper: split by <image> tags as before
    while (!remaining_text.empty()) {
        std::string image_unique_identifier_start = "<" + _image_unique_identifier + ">";
        std::string image_unique_identifier_end = "</" + _image_unique_identifier + ">";
        size_t image_start = remaining_text.find(image_unique_identifier_start);
        if (image_start != std::string::npos) {
            size_t image_end = remaining_text.find(image_unique_identifier_end, image_start);
            if (image_end != std::string::npos) {
                // Add text before image tag as a regular chunk if not empty
                if (image_start > 0) {
                    std::string before_image = remaining_text.substr(0, image_start);
                    if (!before_image.empty()) {
                        auto tokens = _models.at(model_id)->tokenizer->encode(before_image);
                        chunks.push_back({tokens, false, ""});
                    }
                }

                // Extract image path and encode the full image tag as tokens
                size_t path_start = image_start + image_unique_identifier_start.length();
                std::string image_path = remaining_text.substr(path_start, image_end - path_start);
                std::string full_image_tag = remaining_text.substr(image_start, image_end + image_unique_identifier_end.length() - image_start); // include both tags
                auto image_tokens = _models.at(model_id)->tokenizer->encode(full_image_tag);
                chunks.push_back({image_tokens, true, image_path});

                // Update remaining text
                remaining_text = remaining_text.substr(image_end + image_unique_identifier_end.length());
                continue;
            }
        }

        // No more image tags found, process remaining text
        if (!remaining_text.empty()) {
            auto tokens = _models.at(model_id)->tokenizer->encode(remaining_text);
            chunks.push_back({tokens, false, ""});
            break;
        }
    }

    // Now split regular token chunks by max_tokens_per_chunk, also split at "Assistant" and " think"
    std::vector<TokenChunk> final_chunks;
    const std::string assistant_str = "Assistant";
    const std::string thinking_str = " think";
    for (const auto& chunk : chunks) {
        if (chunk.is_image) {
            final_chunks.push_back(chunk);
        } else {
            // Decode tokens to string for splitting
            const std::vector<int>& tokens_to_split = chunk.tokens;
            if (tokens_to_split.empty()) continue;
            std::string chunk_text = _models.at(model_id)->tokenizer->decode(tokens_to_split);

            // First split by "Assistant"
            size_t pos = 0;
            size_t last_pos = 0;
            std::vector<std::string> split_texts;
            std::vector<bool> assistant_belongs;
            while ((pos = chunk_text.find(assistant_str, last_pos)) != std::string::npos) {
                // non-empty part before "Assistant"
                if (pos > last_pos) {
                    split_texts.push_back(chunk_text.substr(last_pos, pos - last_pos));
                    assistant_belongs.push_back(false);
                }
                // "Assistant" chunk, mark as belonging
                split_texts.push_back(assistant_str);
                assistant_belongs.push_back(true);
                last_pos = pos + assistant_str.length();
            }
            if (last_pos < chunk_text.size()) {
                split_texts.push_back(chunk_text.substr(last_pos));
                assistant_belongs.push_back(false);
            }

            // For each split chunk, further split by " think"
            std::vector<std::string> further_split_texts;
            std::vector<bool> trigger_belongs; // true: "Assistant" or " think", to be carried; false: normal text
            for (size_t i = 0; i < split_texts.size(); ++i) {
                if (assistant_belongs[i]) {
                    further_split_texts.push_back(split_texts[i]);
                    trigger_belongs.push_back(true);
                    continue;
                }
                // Split current string at every " think"
                size_t tpos = 0, tlast_pos = 0;
                while ((tpos = split_texts[i].find(thinking_str, tlast_pos)) != std::string::npos) {
                    // Before " think"
                    if (tpos > tlast_pos) {
                        further_split_texts.push_back(split_texts[i].substr(tlast_pos, tpos - tlast_pos));
                        trigger_belongs.push_back(false);
                    }
                    // " think" itself
                    further_split_texts.push_back(thinking_str);
                    trigger_belongs.push_back(true);
                    tlast_pos = tpos + thinking_str.length();
                }
                // Remaining part
                if (tlast_pos < split_texts[i].size()) {
                    further_split_texts.push_back(split_texts[i].substr(tlast_pos));
                    trigger_belongs.push_back(false);
                }
            }

            // Now, for each part, encode and split by max_tokens_per_chunk
            std::vector<int> carry_tokens;
            for (size_t i = 0; i < further_split_texts.size(); ++i) {
                std::string part = further_split_texts[i];
                if (trigger_belongs[i]) {
                    // "Assistant" or " think", carry forward
                    auto tokens = _models.at(model_id)->tokenizer->encode(part);
                    carry_tokens.insert(carry_tokens.end(), tokens.begin(), tokens.end());
                    continue;
                }
                // Normal part
                auto part_tokens = _models.at(model_id)->tokenizer->encode(part);
                // Prepend any carried tokens
                if (!carry_tokens.empty()) {
                    part_tokens.insert(part_tokens.begin(), carry_tokens.begin(), carry_tokens.end());
                    carry_tokens.clear();
                }
                // Split by max_tokens_per_chunk
                for (size_t j = 0; j < part_tokens.size(); j += max_tokens_per_chunk) {
                    size_t end = std::min(j + max_tokens_per_chunk, part_tokens.size());
                    std::vector<int> token_chunk(part_tokens.begin() + j, part_tokens.begin() + end);
                    if (!token_chunk.empty()) {
                        final_chunks.push_back({token_chunk, false, ""});
                    }
                }
            }
            // If any "Assistant" or " think" tokens left, put them as a chunk
            if (!carry_tokens.empty()) {
                final_chunks.push_back({carry_tokens, false, ""});
            }
        }
    }

    return final_chunks;
}

int Runtime::save_state_by_history(int model_id, std::vector<std::string> history, std::string state_path) {
    if (_models.find(model_id) == _models.end()) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);
    if (model->backend == nullptr || model->tokenizer == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    if (history.size() % 2 != 0) {
        history.pop_back();
    }

    auto input_text = apply_chat_template(model_id, history, false);
    std::vector<int> text_ids = model->tokenizer->encode(input_text);

    std::vector<int> tokens_to_prefill;
    auto node = model->backend->match_and_load_state(text_ids, tokens_to_prefill);
    auto matched_ids = node->ids;

    LOGI("saving state cache for model \"%s\" and backend: \"%s\" for prefix: \"%s\" ", model->model_path.c_str(), model->backend_name.c_str(), escape_special_chars(model->tokenizer->decode(matched_ids)).c_str());

    std::vector<uint8_t> state_data;
    int ret = model->backend->serialize_runtime_state(node->state, state_data);
    if (ret) {
        LOGE("failed to serialize runtime state\n");
        return ret;
    }

    // Prepare ids data
    const auto& ids_data = matched_ids;
    uint32_t state_size = static_cast<uint32_t>(state_data.size());
    uint32_t ids_size = static_cast<uint32_t>(ids_data.size() * sizeof(int));
    uint32_t logits_size = static_cast<uint32_t>(node->logits.size() * sizeof(float));

    // Write to file
    FILE* f = fopen(state_path.c_str(), "wb");
    if (!f) {
        LOGE("failed to open state_path for writing: %s\n", state_path.c_str());
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_IO;
    }

    size_t write_cnt = 0;
    // Write [state_size_in_bytes]
    write_cnt = fwrite(&state_size, sizeof(uint32_t), 1, f);
    if (write_cnt != 1) {
        LOGE("failed to write state size\n");
        fclose(f);
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_IO;
    }
    // Write [state_data]
    if (!state_data.empty()) {
        write_cnt = fwrite(state_data.data(), 1, state_data.size(), f);
        if (write_cnt != state_data.size()) {
            LOGE("failed to write state data\n");
            fclose(f);
            return RWKV_ERROR_RUNTIME | RWKV_ERROR_IO;
        }
    }
    // Write [ids_size_in_bytes]
    write_cnt = fwrite(&ids_size, sizeof(uint32_t), 1, f);
    if (write_cnt != 1) {
        LOGE("failed to write ids size\n");
        fclose(f);
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_IO;
    }
    // Write [ids_data]
    if (!ids_data.empty()) {
        write_cnt = fwrite(ids_data.data(), sizeof(int), ids_data.size(), f);
        if (write_cnt != ids_data.size()) {
            LOGE("failed to write ids data\n");
            fclose(f);
            return RWKV_ERROR_RUNTIME | RWKV_ERROR_IO;
        }
    }
    // Write [logits_size_in_bytes]
    write_cnt = fwrite(&logits_size, sizeof(uint32_t), 1, f);
    if (write_cnt != 1) {
        LOGE("failed to write logits size\n");
        fclose(f);
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_IO;
    }
    // Write [logits_data]
    if (!node->logits.empty()) {
        write_cnt = fwrite(node->logits.data(), sizeof(float), node->logits.size(), f);
        if (write_cnt != node->logits.size()) {
            LOGE("failed to write logits data\n");
            fclose(f);
            return RWKV_ERROR_RUNTIME | RWKV_ERROR_IO;
        }
    }

    fclose(f);

    LOGI("State and ids saved successfully to %s (state size: %u bytes, ids count: %zu, ids size: %u bytes)", state_path.c_str(), state_size, ids_data.size(), ids_size);

    return RWKV_SUCCESS;
}

int Runtime::load_history_state_to_memory(int model_id, std::string state_path) {
    if (_models.find(model_id) == _models.end()) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);
    if (model->backend == nullptr || model->tokenizer == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    FILE* f = fopen(state_path.c_str(), "rb");
    if (!f) {
        LOGE("failed to open state_path for reading: %s\n", state_path.c_str());
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_IO;
    }

    uint32_t state_size;
    size_t read_cnt = fread(&state_size, sizeof(uint32_t), 1, f);
    if (read_cnt != 1) {
        LOGE("failed to read state size\n");
        fclose(f);
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_IO;
    }

    std::vector<uint8_t> state_data(state_size);
    read_cnt = fread(state_data.data(), 1, state_size, f);
    if (read_cnt != state_size) {
        LOGE("failed to read state data\n");
        fclose(f);
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_IO;
    }

    uint32_t ids_size;
    read_cnt = fread(&ids_size, sizeof(uint32_t), 1, f);
    if (read_cnt != 1) {
        LOGE("failed to read ids size\n");
        fclose(f);
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_IO;
    }

    std::vector<int> ids_data(ids_size / sizeof(int));
    read_cnt = fread(ids_data.data(), sizeof(int), ids_size / sizeof(int), f);
    if (read_cnt != ids_size / sizeof(int)) {
        LOGE("failed to read ids data\n");
        fclose(f);
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_IO;
    }

    uint32_t logits_size;
    read_cnt = fread(&logits_size, sizeof(uint32_t), 1, f);
    if (read_cnt != 1) {
        LOGE("failed to read logits size\n");
        fclose(f);
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_IO;
    }
    std::vector<float> logits_data(logits_size / sizeof(float));
    read_cnt = fread(logits_data.data(), sizeof(float), logits_size / sizeof(float), f);
    if (read_cnt != logits_size / sizeof(float)) {
        LOGE("failed to read logits data\n");
        fclose(f);
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_IO;
    }

    fclose(f);
    LOGI("loaded state_size: %u bytes, ids_size: %u bytes, logits_size: %u bytes", state_size, ids_size, logits_size);

    std::any runtime_state;
    model->backend->deserialize_runtime_state(state_data, runtime_state);

    std::vector<int> tokens_to_prefill;
    auto node = model->backend->match_and_load_state(ids_data, tokens_to_prefill);
    Tensor1D logits_tensor = Tensor1D::make(logits_data.data(), TensorDType::F32, (size_t)model->backend->get_num_vocab());
    model->backend->register_state_checkpoint_with_state(node, tokens_to_prefill, logits_tensor, runtime_state);
    LOGI("loaded state from disk for text: \"%s\"", escape_special_chars(model->tokenizer->decode(node->ids)).c_str());

    return RWKV_SUCCESS;
}

std::string Runtime::get_state_cache_info(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return "";
    }
    auto &model = _models.at(model_id);
    auto state_root = model->backend->state_root.get();

    std::string state_cache_info;
    std::vector<state_node*> tmp;
    tmp.push_back(state_root);
    // traverse the state tree
    while (!tmp.empty()) {
        auto node = tmp.back();
        tmp.pop_back();
        state_cache_info += "text = \"" + escape_special_chars(model->tokenizer->decode(node->ids)) + "\", remaining lifespan = " + std::to_string(node->activation_count) + "\n";
        for (auto &child : node->children) {
            tmp.push_back(child.get());
        }
    }
    return state_cache_info;
}

int Runtime::chat(int model_id, std::vector<std::string> inputs,
    const int max_length, void (*callback)(const char *, const int, const char *),
    bool enable_reasoning, bool force_reasoning, bool add_generation_prompt, int force_lang, std::vector<std::string> roles_map) {
    if (_models.find(model_id) == _models.end()) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);
    if (model->backend == nullptr || model->tokenizer == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    model->is_generating = true;
    model->stop_signal = false;
    model->response_buffer.clear();
    model->response_buffer_ids.clear();
    model->response_buffer_decoded_tokens = 0;
    model->response_buffer_eos_found = false;

    _clear_speed_samples(*model);

    if (force_lang == 1) {
        LOGI("forcing output language to Chinese\n");
    }

    auto input_text = apply_chat_template(model_id, inputs, enable_reasoning, add_generation_prompt, roles_map);
    LOGD("Applied chat template: \"%s\"\n", input_text.c_str());
    std::vector<int> text_ids = model->tokenizer->encode(input_text);
    std::string debug_msg = "text_ids: ";
    for (auto id : text_ids) {
        debug_msg += std::to_string(id) + " ";
    }
    LOGD("%s\n", debug_msg.c_str());

    Tensor1D logits;
    std::vector<int> tokens_to_prefill;
    auto node = model->backend->match_and_load_state(text_ids, tokens_to_prefill);
    LOGI("matched state cache for prefix: \"%s\"", escape_special_chars(model->tokenizer->decode(node->ids)).c_str());

    if (tokens_to_prefill.size() > 0) {
        _prefill_progress_start(model_id, tokens_to_prefill.size());
        auto text_to_prefill = model->tokenizer->decode(tokens_to_prefill);
        LOGI("new text to prefill: \"%s\"", escape_special_chars(text_to_prefill).c_str());

        // Split text by image tags and token count, then process each chunk
        int checkpoint_interval = _get_prefill_checkpoint_interval((int) tokens_to_prefill.size());
        auto token_chunks = split_text_by_image_and_token_num(text_to_prefill, checkpoint_interval, model_id);

        for (const auto& chunk : token_chunks) {
            if (chunk.is_image) {
                // Handle image chunk - encode the image tag for state matching
                LOGD("Processing image chunk: %s\n", chunk.image_path.c_str());
                if (!chunk.tokens.empty()) {
                    int ret;
#ifdef ENABLE_VISION
                    auto start = std::chrono::high_resolution_clock::now();
                    std::vector<float> embeddings;
                    int n_tokens;
                    if (!model->multimodal_encoder->encode(chunk.image_path, embeddings, n_tokens, model->backend->embedding_input_force_no_ln0())) {
                        model->is_generating = false;
                        LOGE("failed to encode image for image chunk\n");
                        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
                    }
                    auto end = std::chrono::high_resolution_clock::now();
                    LOGI("siglip duration: %lld ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
                    ret = eval_logits_with_embeddings(model_id, embeddings.data(), n_tokens, logits);
                    if (ret) {
                        model->is_generating = false;
                        LOGE("failed to eval logits with embeddings for image chunk\n");
                        return ret;
                    }
#endif
                    ret = model->backend->register_state_checkpoint(node, chunk.tokens, logits);
                    if (ret) {
                        model->is_generating = false;
                        LOGE("failed to register state checkpoint for image chunk\n");
                        return ret;
                    }
                    LOGI("registered state for text: \"%s\"", escape_special_chars(model->tokenizer->decode(node->ids)).c_str());
                }
            } else {
                if (chunk.tokens.empty()) {
                    continue;
                }

                int ret = eval_logits(model_id, chunk.tokens, logits);
                if (ret) {
                    model->is_generating = false;
                    LOGE("failed to eval logits\n");
                    return ret;
                }
                ret = model->backend->register_state_checkpoint(node, chunk.tokens, logits);
                if (ret) {
                    model->is_generating = false;
                    LOGE("failed to register state checkpoint\n");
                    return ret;
                }
                LOGI("registered state for text: \"%s\"", escape_special_chars(model->tokenizer->decode(node->ids)).c_str());
            }
        }
    }
    _prefill_progress_finish(model_id);

    std::vector<int> response_ids_raw;

    bool history_ends_with_user_message = inputs.size() % 2 != 0;
    if (roles_map.size() == inputs.size()) {
        history_ends_with_user_message = roles_map.back() == model->user_role;
    }
    std::string role_for_parsing;
    if (!add_generation_prompt) {
        role_for_parsing = !history_ends_with_user_message ? model->response_role : model->user_role;
    } else {
        role_for_parsing = history_ends_with_user_message ? model->response_role : model->user_role;
    }
    model->response_buffer = input_text.substr(input_text.rfind(role_for_parsing + ":") + (role_for_parsing + ":").size());
    model->response_buffer_ids = model->tokenizer->encode(model->response_buffer);
    model->response_buffer_decoded_tokens = (int)model->response_buffer_ids.size();
    int ret;

    model->sampler->clear_occurences();
    if (!inputs.empty()) {
        auto normalize_role = [&](const std::string &role) -> std::string {
            if (role == "user") {
                return model->user_role;
            }
            if (role == "assistant") {
                return model->response_role;
            }
            if (role == "system") {
                return model->system_role;
            }
            return role;
        };
        std::vector<std::string> resolved_roles;
        if (roles_map.size() == inputs.size()) {
            resolved_roles = roles_map;
        } else {
            resolved_roles.resize(inputs.size());
            for (size_t i = 0; i < inputs.size(); i++) {
                resolved_roles[i] = (i % 2 == 0) ? "user" : "assistant";
            }
        }
        std::string last_role = normalize_role(resolved_roles.back());
        if (last_role == model->response_role) {
            std::vector<int> ids = model->tokenizer->encode(" " + inputs[inputs.size() - 1]);
            for (auto id: ids) {
                model->sampler->update_occurences(id);
            }
        }
    }

    if (logits.data_ptr == nullptr) {
        if (!node->logits.empty()) {
            logits = Tensor1D::make(node->logits.data(), TensorDType::F32, (size_t)model->backend->get_num_vocab()).copy();
        } else {
            LOGE("no logits found, neither from saved state nor from new tokens to prefill\n");
            // this should never happen
            return RWKV_ERROR_RUNTIME;
        }
    }

    int decoded_idx = 0;
    bool thinking_end_tag_found = false;
    bool is_pseudo_thinking = enable_reasoning && model->response_buffer.find("</think>") != std::string::npos;
    bool first_token_ban_thinking_tag = !enable_reasoning || is_pseudo_thinking || force_reasoning;

    for (int i = 0; i < max_length; i++) {
        model->sampler->apply_penalties(logits, model->backend->get_num_vocab());

        if (i <= 2 && first_token_ban_thinking_tag) {
            mask_thinking_tag(logits);
        }

        if (i <= 2 && force_lang == 1) {
            mask_non_chinese_tokens(logits, model->backend->get_num_vocab());
        }

        decoded_idx = model->sampler->sample(logits, model->backend->get_num_vocab());
        if (decoded_idx == 0) {
            LOGD("sampled token 0, stopping generation; eval one more step with EOS for state cache\n");
            int eos_token_id = model->tokenizer->encode(model->eos_token)[0];
            ret = eval_logits(model_id, eos_token_id, logits);
            if (ret) {
                model->is_generating = false;
                LOGE("failed to eval logits with EOS\n");
                return ret;
            }
            response_ids_raw.emplace_back(eos_token_id);
            break;
        }

        auto tmp_tokens = response_ids_raw;
        tmp_tokens.emplace_back(decoded_idx);
        auto compare_token_seq = [&](const std::vector<int> token_seq) -> bool {
            if (tmp_tokens.size() < token_seq.size()) {
                return false;
            }
            for (size_t i = 0; i < token_seq.size(); i++) {
                if (tmp_tokens[tmp_tokens.size() - token_seq.size() + i] != token_seq[i]) {
                    return false;
                }
            }
            return true;
        };

        for (auto &stop_code : model->stop_token_seqs) {
            if (enable_reasoning && !thinking_end_tag_found && stop_code[0] == 261) { // \n\n
                continue;
            }
            if (compare_token_seq(stop_code)) {
                LOGD("stop code found: %s\n", escape_special_chars(model->tokenizer->decode(stop_code)).c_str());
                model->response_buffer_eos_found = true;
                break;
            }
        }
        if (enable_reasoning && !thinking_end_tag_found) {
            if (compare_token_seq({61, 48, 35762, 63})) {
                thinking_end_tag_found = true;
            }
        }

        if (model->response_buffer_eos_found || model->stop_signal) {
            LOGD("stopping generation, eos_found: %d, stop_signal: %d\n", model->response_buffer_eos_found, model->stop_signal);
            break;
        }

        ret = eval_logits(model_id, decoded_idx, logits);
        if (ret) {
            model->is_generating = false;
            LOGE("failed to eval logits\n");
            return ret;
        }

        response_ids_raw.emplace_back(decoded_idx);
        model->response_buffer_ids.emplace_back(decoded_idx);

        if (callback) {
            std::string decoded = model->tokenizer->decode(decoded_idx);
            model->response_buffer += decoded;
            model->response_buffer_decoded_tokens = (int)model->response_buffer_ids.size();
            if (callback) {
                callback(model->response_buffer.c_str(), decoded_idx, decoded.c_str());
            }
        }
        if (model->response_buffer.size() > 0 && model->response_buffer[0] == ' ') {
            model->response_buffer = model->response_buffer.substr(1);
        }

        model->sampler->update_occurences(decoded_idx);
    }

    if (response_ids_raw.size() > 0 && max_length > 0) {
        int ret;
        ret = model->backend->register_state_checkpoint(node, response_ids_raw, logits);
        if (ret) {
            model->is_generating = false;
            LOGE("failed to register state checkpoint\n");
            return ret;
        }
        LOGI("registered state for text: \"%s\"", escape_special_chars(model->tokenizer->decode(node->ids)).c_str());
        model->response_buffer = remove_endl(model->response_buffer);
        model->response_buffer = remove_ending_char(model->response_buffer, '\x17');
    }

    model->is_generating = false;
    model->stop_signal = false;

    model->backend->cleanup_state_tree();

    return RWKV_SUCCESS;
}

int Runtime::chat_batch(int model_id, std::vector<std::vector<std::string>> inputs,
    const int max_length, const int batch_size,
    void (*callback_batch)(const int, const char **, const int*, const char **),
    bool enable_reasoning, bool force_reasoning, bool add_generation_prompt,
    int force_lang, std::vector<std::vector<std::string>> roles_map) {
    if (_models.find(model_id) == _models.end()) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);
    if (model->backend == nullptr || model->tokenizer == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    if (force_lang == 1) {
        LOGI("forcing output language to Chinese\n");
    }

    auto &supported_sizes = model->backend->supported_batch_sizes;
    if (std::find(supported_sizes.begin(), supported_sizes.end(), batch_size) == supported_sizes.end()) {
        LOGE("chat_batch: batch size %d is not supported\n", batch_size);
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_UNSUPPORTED;
    }

    if (inputs.size() != batch_size) {
        LOGE("chat_batch: inputs size %d is not equal to batch size %d\n", inputs.size(), batch_size);
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    model->is_generating = true;
    model->stop_signal = false;

    model->response_buffer_batch.resize(batch_size);
    model->response_buffer_ids_batch.resize(batch_size);
    model->response_buffer_decoded_tokens_batch.resize(batch_size);
    model->response_buffer_eos_found_batch.resize(batch_size);
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        model->response_buffer_batch[batch_idx] = "";
        model->response_buffer_ids_batch[batch_idx].clear();
        model->response_buffer_decoded_tokens_batch[batch_idx] = 0;
        model->response_buffer_eos_found_batch[batch_idx] = false;
    }

    _clear_speed_samples(*model);

    std::vector<std::vector<int>> response_ids_raw_batch(batch_size);

    std::vector<std::string> input_texts(batch_size);
    std::vector<std::vector<int>> text_ids_batch(batch_size);
    Tensor1D logits;
    std::vector<state_node*> nodes_batch(batch_size);

    std::vector<std::map<int, float>> occurences_batch(batch_size);
    std::vector<int> decoded_idx(batch_size);
    std::vector<bool> is_pseudo_thinking_batch(batch_size, false);
    std::vector<std::any> state_batch(batch_size);
    std::vector<bool> thinking_end_tag_found_batch(batch_size, false);
    std::vector<std::vector<float>> prefill_logits_f32_batch(batch_size);
    std::vector<std::vector<float>> logits_final_f32_batch(batch_size);
    std::vector<bool> logits_final_set(batch_size, false);

    auto copy_logits_to_f32 = [&](const Tensor1D &t, std::vector<float> &out, int expected_elems) -> int {
        if (t.data_ptr == nullptr || t.count < (size_t)expected_elems) {
            return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
        }
        out.resize((size_t)expected_elems);
        if (t.dtype == TensorDType::F32) {
            std::copy_n(reinterpret_cast<const float*>(t.data_ptr), expected_elems, out.data());
            return RWKV_SUCCESS;
        }
        if (t.dtype == TensorDType::F16) {
            const half_float::half *h = reinterpret_cast<const half_float::half *>(t.data_ptr);
            for (int i = 0; i < expected_elems; ++i) out[(size_t)i] = (float)h[i];
            return RWKV_SUCCESS;
        }
        return RWKV_ERROR_UNSUPPORTED;
    };

    int ret;
    auto num_vocab = model->backend->get_num_vocab();

    std::vector<float> batched_logits_storage((size_t)num_vocab * batch_size);

    // prefill for each batch
    auto normalize_role = [&](const std::string &role) -> std::string {
        if (role == "user") {
            return model->user_role;
        }
        if (role == "assistant") {
            return model->response_role;
        }
        if (role == "system") {
            return model->system_role;
        }
        return role;
    };

    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        auto &input = inputs[batch_idx];
        std::vector<std::string> batch_roles;
        if (roles_map.size() == inputs.size() && roles_map[batch_idx].size() == input.size()) {
            batch_roles = roles_map[batch_idx];
        }
        input_texts[batch_idx] = apply_chat_template(model_id, input, enable_reasoning, add_generation_prompt, batch_roles);
        LOGD("Applied chat template for batch %d: \"%s\"\n", batch_idx, input_texts[batch_idx].c_str());
        text_ids_batch[batch_idx] = model->tokenizer->encode(input_texts[batch_idx]);

        bool history_ends_with_user_message = input.size() % 2 != 0;
        if (batch_roles.size() == input.size()) {
            history_ends_with_user_message = batch_roles.back() == model->user_role;
        }
        std::string role_for_parsing;
        if (!add_generation_prompt) {
            role_for_parsing = !history_ends_with_user_message ? model->response_role : model->user_role;
        } else {
            role_for_parsing = history_ends_with_user_message ? model->response_role : model->user_role;
        }
        model->response_buffer_batch[batch_idx] = input_texts[batch_idx].substr(input_texts[batch_idx].rfind(role_for_parsing + ":") + (role_for_parsing + ":").size());;
        model->response_buffer_ids_batch[batch_idx].clear();
        model->response_buffer_eos_found_batch[batch_idx] = false;
        std::vector<int> tokens_to_prefill;
        nodes_batch[batch_idx] = model->backend->match_and_load_state(text_ids_batch[batch_idx], tokens_to_prefill);
        LOGI("batch %d matched state cache for prefix: \"%s\"", batch_idx, escape_special_chars(model->tokenizer->decode(nodes_batch[batch_idx]->ids)).c_str());

        Tensor1D batch_prefill_logits;

        // prefill needed tokens
        if (tokens_to_prefill.size() > 0) {
            _prefill_progress_start(model_id, tokens_to_prefill.size());
            LOGI("new text to prefill: \"%s\"", escape_special_chars(model->tokenizer->decode(tokens_to_prefill)).c_str());

            int checkpoint_interval = _get_prefill_checkpoint_interval((int) tokens_to_prefill.size());
            for (int j = 0; j < tokens_to_prefill.size(); j += checkpoint_interval) {
                std::vector<int> tokens_to_prefill_chunk = std::vector<int>(tokens_to_prefill.begin() + j, tokens_to_prefill.begin() + std::min(j + checkpoint_interval, (int)tokens_to_prefill.size()));
                int ret = eval_logits(model_id, tokens_to_prefill_chunk, batch_prefill_logits);
                if (ret) {
                    model->is_generating = false;
                    LOGE("failed to eval logits\n");
                    return ret;
                }
                ret = model->backend->register_state_checkpoint(nodes_batch[batch_idx], tokens_to_prefill_chunk, batch_prefill_logits);
                if (ret) {
                    model->is_generating = false;
                    LOGE("failed to register state checkpoint\n");
                    return ret;
                }
                LOGI("registered state for text: \"%s\"", escape_special_chars(model->tokenizer->decode(nodes_batch[batch_idx]->ids)).c_str());
            }
        }
        _prefill_progress_finish(model_id);

        float* batch_slot = batched_logits_storage.data() + (size_t)batch_idx * num_vocab;
        if (batch_prefill_logits.data_ptr != nullptr) {
            ret = copy_logits_to_f32(batch_prefill_logits, prefill_logits_f32_batch[batch_idx], num_vocab);
            if (ret) {
                model->is_generating = false;
                LOGE("failed to snapshot prefill logits for batch %d\n", batch_idx);
                return ret;
            }
        } else if (!nodes_batch[batch_idx]->logits.empty()) {
            prefill_logits_f32_batch[batch_idx].assign(
                nodes_batch[batch_idx]->logits.data(),
                nodes_batch[batch_idx]->logits.data() + num_vocab);
        } else {
            LOGE("no logits found, neither from saved state nor from new tokens to prefill\n");
            model->is_generating = false;
            return RWKV_ERROR_RUNTIME;
        }
        std::copy_n(prefill_logits_f32_batch[batch_idx].data(), num_vocab, batch_slot);

        is_pseudo_thinking_batch[batch_idx] = !enable_reasoning || (enable_reasoning && model->response_buffer_batch[batch_idx].find("</think>") != std::string::npos);

        model->backend->get_state(state_batch[batch_idx]);
    }

    logits = Tensor1D::make(batched_logits_storage.data(), TensorDType::F32, (size_t)num_vocab * batch_size);

    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        model->backend->set_state_on_batch_slot(batch_idx, state_batch[batch_idx]);
    }

    // dynamic batch size
    int current_batch_size = batch_size;
    std::vector<int> active_batch_indices;
    std::vector<int> original_to_active_mapping(batch_size, -1);
    int eos_token_id = model->tokenizer->encode(model->eos_token)[0];

    for (int j = 0; j < batch_size; j++) {
        active_batch_indices.push_back(j);
        original_to_active_mapping[j] = j;
    }

    for (int i = 0; i < max_length; i++) {
        for (int j = 0; j < current_batch_size; j++) {
            int original_j = active_batch_indices[j];
            Tensor1D view = tensor1d_subview(logits, (size_t)j * (size_t)num_vocab, (size_t)num_vocab);
            model->sampler->apply_penalties(view, num_vocab, occurences_batch[original_j],
                model->sampler->get_token_banned(), model->sampler->get_presence_penalty(),
                model->sampler->get_frequency_penalty(), model->sampler->get_penalty_decay());

            if ((is_pseudo_thinking_batch[original_j] || force_reasoning) && i <= 2) {
                mask_thinking_tag(view);
            }

            if (force_lang == 1 && i <= 2) {
                mask_non_chinese_tokens(view, num_vocab);
            }
        }

        decoded_idx = model->sampler->sample_batch(logits, model->backend->get_num_vocab(), model->backend->get_num_vocab(), current_batch_size);

        for (int j = 0; j < current_batch_size; j++) {
            int original_j = active_batch_indices[j];
            if (decoded_idx[j] == 0) {
                LOGD("sampled token 0 for batch %d; will eval with EOS for state cache\n", original_j);
                model->response_buffer_eos_found_batch[original_j] = true;
                if (!logits_final_set[original_j] && i == 0) {
                    logits_final_f32_batch[original_j] = prefill_logits_f32_batch[original_j];
                    logits_final_set[original_j] = true;
                }
            }

            auto tmp_tokens = response_ids_raw_batch[original_j];
            tmp_tokens.emplace_back(decoded_idx[j]);
            auto compare_token_seq = [&](const std::vector<int> token_seq) -> bool {
                if (tmp_tokens.size() < token_seq.size()) {
                    return false;
                }
                for (size_t k = 0; k < token_seq.size(); k++) {
                    if (tmp_tokens[tmp_tokens.size() - token_seq.size() + k] != token_seq[k]) {
                        return false;
                    }
                }
                return true;
            };

            if (!model->response_buffer_eos_found_batch[original_j]) {
                for (auto &stop_code : model->stop_token_seqs) {
                    if (enable_reasoning && !thinking_end_tag_found_batch[original_j] && stop_code[0] == 261) { // \n\n
                        continue;
                    }
                    if (compare_token_seq(stop_code)) {
                        LOGD("stop code found for batch %d: %s\n", original_j, escape_special_chars(model->tokenizer->decode(stop_code)).c_str());
                        model->response_buffer_eos_found_batch[original_j] = true;
                        std::any state_end;
                        model->backend->get_state_on_batch_slot(j, state_end);
                        state_batch[original_j] = std::move(state_end);
                        if (!logits_final_set[original_j]) {
                            if (i == 0) {
                                logits_final_f32_batch[original_j] = prefill_logits_f32_batch[original_j];
                                logits_final_set[original_j] = true;
                            } else {
                                Tensor1D view = tensor1d_subview(logits, (size_t)j * (size_t)num_vocab, (size_t)num_vocab);
                                int r = copy_logits_to_f32(view, logits_final_f32_batch[original_j], num_vocab);
                                if (r) {
                                    model->is_generating = false;
                                    LOGE("failed to snapshot final logits for batch %d\n", original_j);
                                    return r;
                                }
                                logits_final_set[original_j] = true;
                            }
                        }
                        break;
                    }
                }
                if (enable_reasoning && !thinking_end_tag_found_batch[original_j]) {
                    if (compare_token_seq({61, 48, 35762, 63})) {
                        thinking_end_tag_found_batch[original_j] = true;
                    }
                }
            }
        }

        std::vector<int> new_active_batch_indices;
        for (int j = 0; j < batch_size; j++) {
            if (!model->response_buffer_eos_found_batch[j]) {
                new_active_batch_indices.push_back(j);
            }
        }

        int new_active_count = new_active_batch_indices.size();
        if (new_active_count > 0 && new_active_count < current_batch_size) {
            bool new_size_supported = false;
            for (auto size : model->backend->supported_batch_sizes) {
                if (new_active_count == size) {
                    new_size_supported = true;
                    break;
                }
            }

            if (new_size_supported) {
                LOGI("Switching from batch size %d to %d, active batches: ", current_batch_size, new_active_count);
                for (auto idx : new_active_batch_indices) {
                    LOGI("%d ", idx);
                }

                // get state for new active batches
                std::vector<std::any> temp_states(new_active_count);
                for (int k = 0; k < new_active_count; k++) {
                    int original_batch_idx = new_active_batch_indices[k];
                    int current_slot_idx = original_to_active_mapping[original_batch_idx];
                    if (current_slot_idx >= 0) {
                        model->backend->get_state_on_batch_slot(current_slot_idx, temp_states[k]);
                    }
                }

                // rearrange logits for new active batches
                if (logits.dtype != TensorDType::F32 || logits.data_ptr == nullptr) {
                    return RWKV_ERROR_UNSUPPORTED;
                }
                float* logits_f32 = reinterpret_cast<float*>(logits.data_ptr);
                std::vector<float> temp_logits(new_active_count * num_vocab);
                for (int k = 0; k < new_active_count; k++) {
                    int original_batch_idx = new_active_batch_indices[k];
                    int current_slot_idx = original_to_active_mapping[original_batch_idx];
                    if (current_slot_idx >= 0 && current_slot_idx < current_batch_size) {
                        std::copy_n(
                            logits_f32 + current_slot_idx * num_vocab,
                            num_vocab,
                            temp_logits.data() + k * num_vocab
                        );
                    }
                }
                // copy rearranged logits back
                std::copy_n(temp_logits.data(), new_active_count * num_vocab, logits_f32);

                // rearrange decoded_idx for new active batches
                std::vector<int> temp_decoded_idx(new_active_count);
                for (int k = 0; k < new_active_count; k++) {
                    int original_batch_idx = new_active_batch_indices[k];
                    int current_slot_idx = original_to_active_mapping[original_batch_idx];
                    if (current_slot_idx >= 0 && current_slot_idx < current_batch_size) {
                        temp_decoded_idx[k] = decoded_idx[current_slot_idx];
                        LOGD("Rearranging decoded_idx[%d]: original_batch=%d, current_slot=%d, token=%d\n",
                             k, original_batch_idx, current_slot_idx, decoded_idx[current_slot_idx]);
                    } else {
                        LOGE("Invalid mapping for batch %d: current_slot=%d (should be in [0,%d))\n",
                             original_batch_idx, current_slot_idx, current_batch_size);
                        temp_decoded_idx[k] = 0; // fallback to EOS to avoid undefined behavior
                    }
                }
                // copy rearranged decoded_idx back
                for (int k = 0; k < new_active_count; k++) {
                    decoded_idx[k] = temp_decoded_idx[k];
                }

                // rearrange active state to corresponding slots
                for (int k = 0; k < new_active_count; k++) {
                    model->backend->set_state_on_batch_slot(k, temp_states[k]);
                    model->backend->free_state(temp_states[k]);
                }

                // update active batch indices and mapping
                active_batch_indices = new_active_batch_indices;
                current_batch_size = new_active_count;

                original_to_active_mapping.assign(batch_size, -1);
                for (int k = 0; k < new_active_count; k++) {
                    original_to_active_mapping[active_batch_indices[k]] = k;
                }
            }
        }

        auto all_eos_found = std::all_of(model->response_buffer_eos_found_batch.begin(), model->response_buffer_eos_found_batch.end(), [](bool eos_found) { return eos_found; });
        if (all_eos_found || model->stop_signal) {
            LOGD("stopping generation, eos_found: %d, stop_signal: %d\n", all_eos_found, model->stop_signal);
            break;
        }

        std::vector<int> active_decoded_idx(current_batch_size);
        for (int j = 0; j < current_batch_size; j++) {
            active_decoded_idx[j] = (decoded_idx[j] == 0) ? eos_token_id : decoded_idx[j];
        }
        ret = eval_logits_batch_decode(model_id, active_decoded_idx, logits);
        if (ret) {
            model->is_generating = false;
            LOGE("failed to eval logits\n");
            return ret;
        }

        for (int j = 0; j < current_batch_size; j++) {
            int original_j = active_batch_indices[j];
            if (decoded_idx[j] == 0) {
                std::any state_end;
                model->backend->get_state_on_batch_slot(j, state_end);
                state_batch[original_j] = std::move(state_end);
                if (!logits_final_set[original_j]) {
                    Tensor1D view = tensor1d_subview(logits, (size_t)j * (size_t)num_vocab, (size_t)num_vocab);
                    int r = copy_logits_to_f32(view, logits_final_f32_batch[original_j], num_vocab);
                    if (r) {
                        model->is_generating = false;
                        LOGE("failed to snapshot final logits for batch %d\n", original_j);
                        return r;
                    }
                    logits_final_set[original_j] = true;
                }
                response_ids_raw_batch[original_j].emplace_back(eos_token_id);
                continue;
            }
            if (model->response_buffer_eos_found_batch[original_j]) {
                continue;
            }
            response_ids_raw_batch[original_j].emplace_back(decoded_idx[j]);
            model->response_buffer_ids_batch[original_j].emplace_back(decoded_idx[j]);
            if (model->response_buffer_batch[original_j].size() > 0 && model->response_buffer_batch[original_j][0] == ' ') {
                model->response_buffer_batch[original_j] = model->response_buffer_batch[original_j].substr(1);
            }

            occurences_batch[original_j][decoded_idx[j]]++;
        }

        // TODO: callback_batch
        // if (callback_batch) {
        //     callback_batch(...);
        // }
    }

    // save state for not finished batches
    for (int j = 0; j < batch_size; j++) {
        if (!model->response_buffer_eos_found_batch[j]) {
            int active_slot = original_to_active_mapping[j];
            if (active_slot >= 0) {
                model->backend->get_state_on_batch_slot(active_slot, state_batch[j]);
            }
        }
    }

    bool any_generated = false;
    for (int j = 0; j < batch_size; j++) {
        if (!response_ids_raw_batch[j].empty()) {
            any_generated = true;
            break;
        }
    }
    if (any_generated) {
        // Register per-batch checkpoints with correct logits for each original batch.
        // Dynamic batch resizing reorders/overwrites the shared logits buffer, so we must not rely on it directly for finished batches.
        for (int j = 0; j < batch_size; j++) {
            if (response_ids_raw_batch[j].empty()) {
                continue; // no new tokens generated => no new checkpoint needed
            }

            Tensor1D logits_view{};
            if (logits_final_set[j]) {
                logits_view = Tensor1D::make(logits_final_f32_batch[j].data(), TensorDType::F32, (size_t)num_vocab);
            } else {
                int active_slot = original_to_active_mapping[j];
                if (active_slot >= 0) {
                    logits_view = tensor1d_subview(logits, (size_t)active_slot * (size_t)num_vocab, (size_t)num_vocab);
                } else {
                    // Fallback: should be rare (e.g., ended before first batch decode but no final snapshot).
                    logits_view = Tensor1D::make(prefill_logits_f32_batch[j].data(), TensorDType::F32, (size_t)num_vocab);
                }
            }

            int r = model->backend->register_state_checkpoint_with_state(nodes_batch[j], response_ids_raw_batch[j], logits_view, state_batch[j]);
            if (r) {
                model->is_generating = false;
                LOGE("failed to register state checkpoint for batch %d\n", j);
                return r;
            }

            model->response_buffer_batch[j] = remove_endl(model->response_buffer_batch[j]);
            model->response_buffer_batch[j] = remove_ending_char(model->response_buffer_batch[j], '\x17');
        }
    }

    model->is_generating = false;
    model->stop_signal = false;

    model->backend->cleanup_state_tree();

    return RWKV_SUCCESS;
}

int Runtime::prefill_to_cache(int model_id, std::string text) {
    if (_models.find(model_id) == _models.end()) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);
    if (model->backend == nullptr || model->tokenizer == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    std::vector<int> ids = model->tokenizer->encode(text);
    if (ids.empty()) {
        return RWKV_SUCCESS;
    }
    std::vector<int> new_ids_to_prefill;
    auto node = model->backend->match_and_load_state(ids, new_ids_to_prefill);
    if (new_ids_to_prefill.empty()) {
        return RWKV_SUCCESS;
    }

    model->backend->set_state(node->state);

    Tensor1D logits;
    int ret = eval_logits(model_id, ids, logits);
    if (ret) {
        return ret;
    }
    model->backend->register_state_checkpoint(node, ids, logits);
    return RWKV_SUCCESS;
}

int Runtime::set_prompt(int model_id, std::string prompt) {
    if (_models.find(model_id) == _models.end()) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);
    if (model->backend == nullptr || model->tokenizer == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    LOGD("Setting and processing prompt for model %d: \"%s\"\n", model_id, prompt.c_str());
    model->prompt = prompt;
    auto ret = prefill_to_cache(model_id, prompt);
    if (ret) {
        return ret;
    }
    return prefill_to_cache(model_id, prompt + model->bos_token + model->user_role);
}

std::string Runtime::get_prompt(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return "";
    }
    auto &model = _models.at(model_id);
    return model->prompt;
}

#ifdef ENABLE_WHISPER
int Runtime::set_audio_prompt(int model_id, std::string path) {
    if (_models.find(model_id) == _models.end()) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);
    if (model->backend == nullptr || model->tokenizer == nullptr || model->multimodal_encoder == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    std::string prompt = "<audio src=\"" + path + "\">";
    std::vector<int> ids = model->tokenizer->encode(prompt);

    std::vector<int> new_ids_to_prefill;
    auto node = model->backend->match_and_load_state(ids, new_ids_to_prefill);
    if (new_ids_to_prefill.empty()) {
        return RWKV_SUCCESS;
    }

    model->prompt = prompt;
    model->backend->set_state(node->state);

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<float> embeddings;
    int n_tokens;
    if (!model->multimodal_encoder->encode(path, embeddings, n_tokens, model->backend->embedding_input_force_no_ln0())) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto end = std::chrono::high_resolution_clock::now();
    LOGI("whisper duration: %lld ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    Tensor1D logits;

    int ret = eval_logits_with_embeddings(model_id, embeddings.data(), n_tokens, logits);
    if (ret) {
        LOGE("%s failed to eval logits with embeddings\n", __func__);
        return ret;
    }
    model->backend->register_state_checkpoint(node, ids, logits);
    return RWKV_SUCCESS;
}
#endif

#ifdef ENABLE_TTS
namespace {
int generate_tts_output(
    Runtime* rt,
    int model_id,
    NucleusSampler* sampler,
    execution_provider* backend,
    Tensor1D& logits,
    std::vector<int>& output_tokens
) {
    static const int tts_max_length = 3000;
    static const int tts_top_k = 50;
    static const float tts_top_p = 0.95;
    static const float tts_temperature = 1.0;
    static const int tts_eos_token = 8192;
    static const int tts_tag_token_offset = 8193;

    for (int i = 0; i < tts_max_length; i++) {
        int idx = sampler->sample(logits, tts_tag_token_offset, tts_temperature, tts_top_k, tts_top_p);
        if (idx == tts_eos_token) {
            break;
        }

        output_tokens.push_back(idx);
        int ret = rt->eval_logits(model_id, idx, logits);
        if (ret || logits.data_ptr == nullptr) {
            LOGE("[TTS] Error evaluating logits");
            return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
        }
    }
    return RWKV_SUCCESS;
}

void tts_detokenize_thread_main(
    sparktts* sparktts_instance,
    Runtime* rt,
    const std::vector<int>& global_tokens,
    const std::vector<int>& output_tokens,
    const bool& generation_finished,
    double& ttfa,
    const std::chrono::high_resolution_clock::time_point& total_start
) {
    int actual_chunk_size = sparktts_instance->initial_chunk_size;
    std::vector<int> semantic_tokens_buf;
    sparktts_instance->resize_detokenizer_model(actual_chunk_size);
    int semantic_token_pos = 0;
    while (!generation_finished) {
        if (output_tokens.size() - semantic_token_pos >= actual_chunk_size) {
            std::vector<int> current_chunk_tokens(output_tokens.begin() + semantic_token_pos, output_tokens.begin() + semantic_token_pos + actual_chunk_size);
            semantic_token_pos += actual_chunk_size;
            int buffered_size = semantic_tokens_buf.size();
            std::vector<int> current_semantic_tokens = semantic_tokens_buf;
            current_semantic_tokens.insert(current_semantic_tokens.end(), current_chunk_tokens.begin(), current_chunk_tokens.end());
            semantic_tokens_buf = std::vector<int>(current_semantic_tokens.begin() + (current_semantic_tokens.size() - sparktts_instance->overlap_size), current_semantic_tokens.end());
            auto new_samples = sparktts_instance->detokenize_audio(global_tokens, current_semantic_tokens);
            if (rt->tts_get_streaming_buffer().empty()) {
                ttfa = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - total_start).count();
                actual_chunk_size = sparktts_instance->chunk_size;
                sparktts_instance->resize_detokenizer_model(actual_chunk_size);
            }
            // tts_output_samples_buffer.insert(tts_output_samples_buffer.end(), new_samples.begin() + (16000 * buffered_size / 50), new_samples.end());
            rt->tts_append_samples_to_buffer(new_samples.begin() + (16000 * buffered_size / 50), new_samples.end());
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    if (output_tokens.size() - semantic_token_pos > 0) {
        std::vector<int> current_chunk_tokens(output_tokens.begin() + semantic_token_pos, output_tokens.end());
        int buffered_size = semantic_tokens_buf.size();
        std::vector<int> current_semantic_tokens = semantic_tokens_buf;
        current_semantic_tokens.insert(current_semantic_tokens.end(), current_chunk_tokens.begin(), current_chunk_tokens.end());
        auto new_samples = sparktts_instance->detokenize_audio(global_tokens, current_semantic_tokens);
        // tts_output_samples_buffer.insert(tts_output_samples_buffer.end(), new_samples.begin() + (16000 * buffered_size / 50), new_samples.end());
        rt->tts_append_samples_to_buffer(new_samples.begin() + (16000 * buffered_size / 50), new_samples.end());
    }
}
} // anonymous namespace

int Runtime::sparktts_load_models(
    std::string wav2vec2_path,
    std::string bicodec_tokenizer_path,
    std::string bicodec_detokenizer_path
) {
    _sparktts = std::make_unique<sparktts>();
    if (!_sparktts->load_models(wav2vec2_path, bicodec_tokenizer_path, bicodec_detokenizer_path)) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    return RWKV_SUCCESS;
}

int Runtime::sparktts_release_models() {
    _sparktts = nullptr;
    return RWKV_SUCCESS;
}

int Runtime::run_spark_tts_zeroshot(int model_id, std::string tts_text, std::string prompt_audio_text, std::string prompt_audio_path, std::string output_wav_path) {
    if (_sparktts == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);

    static const int tts_tag_token_offset = 8193;
    static const int global_token_offset = 8196;

    wav_file wav;
    wav.load(prompt_audio_path);
    wav.resample(16000);

    auto total_start = std::chrono::high_resolution_clock::now();
    std::vector<int> global_tokens;
    std::vector<int> semantic_tokens;
    _sparktts->tokenize_audio(wav.samples, global_tokens, semantic_tokens);
    if (prompt_audio_text.empty()) {
        semantic_tokens.clear();
    }

    std::string full_text = prompt_audio_text + tts_text;
    auto text_tokens = tokenizer_encode(model_id, full_text);
    if (text_tokens.empty()) {
        LOGE("[TTS] Text tokenizer encode failed");
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    std::vector<int> input_tokens = {tts_tag_token_offset + 2}; // tag_2
    for (int i = 0; i < text_tokens.size(); i++) {
        input_tokens.push_back(text_tokens[i]);
    }
    input_tokens.push_back(tts_tag_token_offset + 0); // tag_0
    for (int i = 0; i < global_tokens.size(); i++) {
        input_tokens.push_back(global_tokens[i] + global_token_offset);
    }
    input_tokens.push_back(tts_tag_token_offset + 1); // tag_1
    for (int i = 0; i < semantic_tokens.size(); i++) {
        input_tokens.push_back(semantic_tokens[i]);
    }

    _global_tokens_output.clear();
    for (auto token : global_tokens) {
        _global_tokens_output.push_back(token + global_token_offset);
    }

    std::vector<int> output_tokens;

    auto start = std::chrono::high_resolution_clock::now();

    clear_state(model_id);
    Tensor1D logits;
    int ret = eval_logits(model_id, input_tokens, logits);
    if (ret || logits.data_ptr == nullptr) {
        LOGE("[TTS] Error evaluating logits");
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    ret = generate_tts_output(this, model_id, model->sampler.get(), model->backend.get(), logits, output_tokens);
    if (ret) return ret;

    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    LOGI("[TTS] LLM inference time: %lf ms", duration);
    LOGI("[TTS] LLM output tokens: %d", output_tokens.size());
    LOGI("[TTS] LLM prefill speed: %f tokens/s", get_avg_prefill_speed(model_id));
    LOGI("[TTS] LLM decode speed: %f tokens/s", get_avg_decode_speed(model_id));

    std::vector<float> output_samples = _sparktts->detokenize_audio(global_tokens, output_tokens);
    save_samples_to_wav(output_samples, output_wav_path, 16000);

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
    LOGI("[TTS] Total time: %lf ms", total_duration);
    LOGI("[TTS] Output audio length: %lf s", output_samples.size() / 16000.0);
    LOGI("[TTS] RTF: %lf", total_duration / 1e3f * 16000.0 / output_samples.size());

    set_is_generating(model_id, false);
    return RWKV_SUCCESS;
}

int Runtime::run_spark_tts_with_properties(int model_id, std::string tts_text, std::string output_wav_path,
    std::string age, std::string gender, std::string emotion, std::string pitch, std::string speed) {
    if (_sparktts == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);

    static const int tts_tag_token_offset = 8193;
    static const int global_token_offset = 8196;

    std::vector<int> properties_tokens = convert_standard_properties_to_tokens(age, gender, emotion, pitch, speed);

    auto total_start = std::chrono::high_resolution_clock::now();
    std::vector<int> global_tokens;
    std::vector<int> semantic_tokens;

    auto text_tokens = tokenizer_encode(model_id, tts_text);
    if (text_tokens.empty()) {
        LOGE("[TTS] Text tokenizer encode failed");
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    std::vector<int> input_tokens = properties_tokens;
    input_tokens.push_back(tts_tag_token_offset + 2); // tag_2
    for (int i = 0; i < text_tokens.size(); i++) {
        input_tokens.push_back(text_tokens[i]);
    }
    input_tokens.push_back(tts_tag_token_offset + 0); // tag_0

    clear_state(model_id);
    Tensor1D logits;
    int ret = eval_logits(model_id, input_tokens, logits);
    if (ret || logits.data_ptr == nullptr) {
        LOGE("[TTS] Error evaluating logits");
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    std::vector<float> logits_scratch;
    for (int i = 0; i < 32; i++) { // generate 32 global_tokens
        int idx = model->sampler->sample(logits, 4096, 1.0, 20, 0.95);

        global_tokens.push_back(idx + global_token_offset);
        ret = eval_logits(model_id, idx + global_token_offset, logits);
        if (ret || logits.data_ptr == nullptr) {
            LOGE("[TTS] Error evaluating logits");
            return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
        }
    }

    _global_tokens_output = global_tokens;

    ret = eval_logits(model_id, tts_tag_token_offset + 1, logits);
    if (ret || logits.data_ptr == nullptr) {
        LOGE("[TTS] Error evaluating logits");
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    std::vector<int> output_tokens;

    ret = generate_tts_output(this, model_id, model->sampler.get(), model->backend.get(), logits, output_tokens);
    if (ret) return ret;

    LOGI("[TTS] LLM output tokens: %d", output_tokens.size());
    LOGI("[TTS] LLM prefill speed: %f tokens/s", get_avg_prefill_speed(model_id));
    LOGI("[TTS] LLM decode speed: %f tokens/s", get_avg_decode_speed(model_id));

    std::vector<float> output_samples = _sparktts->detokenize_audio(global_tokens, output_tokens);
    save_samples_to_wav(output_samples, output_wav_path, 16000);

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
    LOGI("[TTS] Total time: %lf ms", total_duration);
    LOGI("[TTS] Output audio length: %lf s", output_samples.size() / 16000.0);
    LOGI("[TTS] RTF: %lf", total_duration / 1e3f * 16000.0 / output_samples.size());

    set_is_generating(model_id, false);
    return RWKV_SUCCESS;
}

int Runtime::run_spark_tts_with_global_tokens(int model_id, std::string tts_text, std::string output_wav_path, std::vector<int> global_tokens) {
    if (_sparktts == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);

    static const int tts_tag_token_offset = 8193;

    auto total_start = std::chrono::high_resolution_clock::now();

    auto text_tokens = tokenizer_encode(model_id, tts_text);
    if (text_tokens.empty()) {
        LOGE("[TTS] Text tokenizer encode failed");
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    std::vector<int> input_tokens;
    input_tokens.push_back(tts_tag_token_offset + 2); // tag_2
    for (int i = 0; i < text_tokens.size(); i++) {
        input_tokens.push_back(text_tokens[i]);
    }
    input_tokens.push_back(tts_tag_token_offset + 0); // tag_0
    for (int i = 0; i < global_tokens.size(); i++) {
        input_tokens.push_back(global_tokens[i]); // global_tokens are already offset by global_token_offset
    }
    input_tokens.push_back(tts_tag_token_offset + 1); // tag_1

    clear_state(model_id);
    Tensor1D logits;
    int ret = eval_logits(model_id, input_tokens, logits);
    if (ret || logits.data_ptr == nullptr) {
        LOGE("[TTS] Error evaluating logits");
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    std::vector<int> output_tokens;

    ret = generate_tts_output(this, model_id, model->sampler.get(), model->backend.get(), logits, output_tokens);
    if (ret) return ret;

    LOGI("[TTS] LLM output tokens: %d", output_tokens.size());
    LOGI("[TTS] LLM prefill speed: %f tokens/s", get_avg_prefill_speed(model_id));
    LOGI("[TTS] LLM decode speed: %f tokens/s", get_avg_decode_speed(model_id));

    std::vector<float> output_samples = _sparktts->detokenize_audio(global_tokens, output_tokens);
    save_samples_to_wav(output_samples, output_wav_path, 16000);

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
    LOGI("[TTS] Total time: %lf ms", total_duration);
    LOGI("[TTS] Output audio length: %lf s", output_samples.size() / 16000.0);
    LOGI("[TTS] RTF: %lf", total_duration / 1e3f * 16000.0 / output_samples.size());

    set_is_generating(model_id, false);
    return RWKV_SUCCESS;
}

int Runtime::run_spark_tts_zeroshot_streaming(int model_id, std::string tts_text, std::string prompt_audio_text, std::string prompt_audio_path, std::string output_wav_path) {
    if (_sparktts == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

#if !defined(_WIN32)
    auto texts = tts_frontend_utils::process_text(tts_text,
        [this, model_id](const std::string& text) -> std::vector<int> {
            return tokenizer_encode(model_id, text);
        },
        _tn_list
    );
#else
    auto texts = tts_frontend_utils::process_text(tts_text,
        [this, model_id](const std::string& text) -> std::vector<int> {
            return tokenizer_encode(model_id, text);
        }
    );
#endif

    tts_clear_streaming_buffer();
    auto total_start = std::chrono::high_resolution_clock::now();
    std::vector<int> global_tokens;
    std::vector<int> semantic_tokens;

    bool read_from_cache = _sparktts->get_global_and_semantic_tokens(prompt_audio_path, _cache_dir, global_tokens, semantic_tokens);
    if (semantic_tokens.empty() || global_tokens.empty()) {
        LOGE("[TTS] Failed to get global and/or semantic tokens");
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    static const int global_token_offset = 8196;
    _global_tokens_output.clear();
    for (auto token : global_tokens) {
        _global_tokens_output.push_back(token + global_token_offset);
    }

    if (prompt_audio_text.empty()) {
        semantic_tokens.clear();
    }

    // variables between threads
    bool generation_finished = false;
    std::vector<int> output_tokens;

    auto llm_inference_thread = std::thread([&]() {
        auto &model = _models.at(model_id);
        static const int tts_tag_token_offset = 8193;
        static const int global_token_offset = 8196;

        static const int tts_max_length = 3000;
        static const int tts_top_k = 50;
        static const float tts_top_p = 0.95;
        static const float tts_temperature = 1.0;
        static const int tts_eos_token = 8192;
        for (auto &text : texts) {
            std::string full_text = prompt_audio_text + text;
            LOGI("[TTS] LLM input text: %s", full_text.c_str());
            auto text_tokens = tokenizer_encode(model_id, full_text);
            if (text_tokens.empty()) {
                LOGE("[TTS] Text tokenizer encode failed");
                generation_finished = true;
                return;
            }

            std::vector<int> input_tokens = {tts_tag_token_offset + 2}; // tag_2
            for (int i = 0; i < text_tokens.size(); i++) {
                input_tokens.push_back(text_tokens[i]);
            }
            input_tokens.push_back(tts_tag_token_offset + 0); // tag_0
            for (int i = 0; i < global_tokens.size(); i++) {
                input_tokens.push_back(global_tokens[i] + global_token_offset);
            }
            input_tokens.push_back(tts_tag_token_offset + 1); // tag_1
            for (int i = 0; i < semantic_tokens.size(); i++) {
                input_tokens.push_back(semantic_tokens[i]);
            }

            clear_state(model_id);
            Tensor1D logits;
            int ret = eval_logits(model_id, input_tokens, logits);
            if (ret || logits.data_ptr == nullptr) {
                LOGE("[TTS] Error evaluating logits");
                generation_finished = true;
                return;
            }
            tensor1d_set_f32(logits, (size_t)tts_eos_token, -1e9f);

            for (int i = 0; i < tts_max_length; i++) {
                int idx = model->sampler->sample(logits, tts_tag_token_offset, tts_temperature, tts_top_k, tts_top_p);
                if (idx == tts_eos_token) {
                    LOGI("[TTS] EOS token found");
                    break;
                }

                output_tokens.push_back(idx);
                ret = eval_logits(model_id, idx, logits);
                if (ret || logits.data_ptr == nullptr) {
                    LOGE("[TTS] Error evaluating logits");
                    generation_finished = true;
                    return;
                }
            }
        }
        generation_finished = true;
    });

    double ttfa = 0.0;
    std::thread detokenize_thread([&]() {
        tts_detokenize_thread_main(_sparktts.get(), this, global_tokens, output_tokens, generation_finished, ttfa, total_start);
    });

    llm_inference_thread.join();
    detokenize_thread.join();

    LOGI("[TTS] LLM output tokens: %d", output_tokens.size());
    LOGI("[TTS] LLM prefill speed: %f tokens/s", get_avg_prefill_speed(model_id));
    LOGI("[TTS] LLM decode speed: %f tokens/s", get_avg_decode_speed(model_id));
    if (!_tts_output_samples_buffer.empty()) {
        save_samples_to_wav(_tts_output_samples_buffer, output_wav_path, 16000);
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
    LOGI("[TTS] Total time (%s): %lf ms", read_from_cache ? "prompt audio tokens cache hit" : "prompt audio tokens cache miss", total_duration);
    LOGI("[TTS] Output audio length: %lf s", _tts_output_samples_buffer.size() / 16000.0);
    LOGI("[TTS] RTF (%s): %lf", read_from_cache ? "prompt audio tokens cache hit" : "prompt audio tokens cache miss", total_duration / 1e3f * 16000.0 / _tts_output_samples_buffer.size());
    LOGI("[TTS] TTFA (%s): %lf ms", read_from_cache ? "prompt audio tokens cache hit" : "prompt audio tokens cache miss", ttfa);

    set_is_generating(model_id, false);
    return RWKV_SUCCESS;
}

int Runtime::run_spark_tts_with_properties_streaming(int model_id, std::string tts_text, std::string output_wav_path,
    std::string age, std::string gender, std::string emotion, std::string pitch, std::string speed) {
    if (_sparktts == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

#if !defined(_WIN32)
    auto texts = tts_frontend_utils::process_text(tts_text,
        [this, model_id](const std::string& text) -> std::vector<int> {
            return tokenizer_encode(model_id, text);
        },
        _tn_list
    );
#else
    auto texts = tts_frontend_utils::process_text(tts_text,
        [this, model_id](const std::string& text) -> std::vector<int> {
            return tokenizer_encode(model_id, text);
        }
    );
#endif

    tts_clear_streaming_buffer();
    auto total_start = std::chrono::high_resolution_clock::now();
    std::vector<int> global_tokens;
    std::vector<int> properties_tokens = convert_standard_properties_to_tokens(age, gender, emotion, pitch, speed);

    // variables between threads
    bool generation_finished = false;
    std::vector<int> output_tokens;

    auto llm_inference_thread = std::thread([&]() {
        auto &model = _models.at(model_id);
        static const int tts_tag_token_offset = 8193;
        static const int global_token_offset = 8196;

        static const int tts_max_length = 3000;
        static const int tts_top_k = 50;
        static const float tts_top_p = 0.95;
        static const float tts_temperature = 1.0;
        static const int tts_eos_token = 8192;
        for (auto &text : texts) {
            LOGI("[TTS] LLM input text: %s", text.c_str());
            auto text_tokens = tokenizer_encode(model_id, text);
            if (text_tokens.empty()) {
                LOGE("[TTS] Text tokenizer encode failed");
                generation_finished = true;
                return;
            }

            clear_state(model_id);
            Tensor1D logits;

            if (global_tokens.empty()) {
                // generate global tokens
                std::vector<int> input_tokens = properties_tokens;
                input_tokens.push_back(tts_tag_token_offset + 2); // tag_2
                for (int i = 0; i < text_tokens.size(); i++) {
                    input_tokens.push_back(text_tokens[i]);
                }
                input_tokens.push_back(tts_tag_token_offset + 0); // tag_0

                int ret = eval_logits(model_id, input_tokens, logits);
                if (ret || logits.data_ptr == nullptr) {
                    LOGE("[TTS] Error evaluating logits");
                    generation_finished = true;
                    return;
                }

                for (int i = 0; i < 32; i++) { // generate 32 global_tokens
                    int idx = model->sampler->sample(logits, 4096, 1.0, 20, 0.95);

                    global_tokens.push_back(idx + global_token_offset);
                    ret = eval_logits(model_id, idx + global_token_offset, logits);
                    if (ret || logits.data_ptr == nullptr) {
                        LOGE("[TTS] Error evaluating logits");
                        generation_finished = true;
                        return;
                    }
                }

                _global_tokens_output = global_tokens;

                ret = eval_logits(model_id, tts_tag_token_offset + 1, logits);
                if (ret || logits.data_ptr == nullptr) {
                    LOGE("[TTS] Error evaluating logits");
                    generation_finished = true;
                    return;
                }
            } else {
                std::vector<int> input_tokens = {tts_tag_token_offset + 2}; // tag_2
                for (int i = 0; i < text_tokens.size(); i++) {
                    input_tokens.push_back(text_tokens[i]);
                }
                input_tokens.push_back(tts_tag_token_offset + 0); // tag_0
                for (int i = 0; i < global_tokens.size(); i++) {
                    input_tokens.push_back(global_tokens[i] + global_token_offset);
                }
                input_tokens.push_back(tts_tag_token_offset + 1); // tag_1

                int ret = eval_logits(model_id, input_tokens, logits);
                if (ret || logits.data_ptr == nullptr) {
                    LOGE("[TTS] Error evaluating logits");
                    generation_finished = true;
                    return;
                }
            }

            tensor1d_set_f32(logits, (size_t)tts_eos_token, -1e9f);

            for (int i = 0; i < tts_max_length; i++) {
                int idx = model->sampler->sample(logits, tts_tag_token_offset, tts_temperature, tts_top_k, tts_top_p);
                if (idx == tts_eos_token) {
                    LOGI("[TTS] EOS token found");
                    break;
                }

                output_tokens.push_back(idx);
                int ret = eval_logits(model_id, idx, logits);
                if (ret || logits.data_ptr == nullptr) {
                    LOGE("[TTS] Error evaluating logits");
                    generation_finished = true;
                    return;
                }
            }
        }
        generation_finished = true;
    });

    double ttfa = 0.0;
    std::thread detokenize_thread([&]() {
        tts_detokenize_thread_main(_sparktts.get(), this, global_tokens, output_tokens, generation_finished, ttfa, total_start);
    });

    llm_inference_thread.join();
    detokenize_thread.join();

    LOGI("[TTS] %s: LLM output tokens: %d", __func__, output_tokens.size());
    LOGI("[TTS] %s: LLM prefill speed: %f tokens/s", __func__, get_avg_prefill_speed(model_id));
    LOGI("[TTS] %s: LLM decode speed: %f tokens/s", __func__, get_avg_decode_speed(model_id));
    if (!_tts_output_samples_buffer.empty()) {
        save_samples_to_wav(_tts_output_samples_buffer, output_wav_path, 16000);
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
    LOGI("[TTS] %s: Total time: %lf ms", __func__, total_duration);
    LOGI("[TTS] %s: Output audio length: %lf s", __func__, _tts_output_samples_buffer.size() / 16000.0);
    LOGI("[TTS] %s: RTF: %lf", __func__, total_duration / 1e3f * 16000.0 / _tts_output_samples_buffer.size());
    LOGI("[TTS] %s: TTFA: %lf ms", __func__, ttfa);

    set_is_generating(model_id, false);
    return RWKV_SUCCESS;
}


int Runtime::run_spark_tts_with_global_tokens_streaming(int model_id, std::string tts_text, std::string output_wav_path, std::vector<int> global_tokens) {
    if (_sparktts == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

#if !defined(_WIN32)
    auto texts = tts_frontend_utils::process_text(tts_text,
        [this, model_id](const std::string& text) -> std::vector<int> {
            return tokenizer_encode(model_id, text);
        },
        _tn_list
    );
#else
    auto texts = tts_frontend_utils::process_text(tts_text,
        [this, model_id](const std::string& text) -> std::vector<int> {
            return tokenizer_encode(model_id, text);
        }
    );
#endif

    tts_clear_streaming_buffer();
    auto total_start = std::chrono::high_resolution_clock::now();

    // variables between threads
    bool generation_finished = false;
    std::vector<int> output_tokens;

    auto llm_inference_thread = std::thread([&]() {
        auto &model = _models.at(model_id);
        static const int tts_tag_token_offset = 8193;
        static const int global_token_offset = 8196;

        static const int tts_max_length = 3000;
        static const int tts_top_k = 50;
        static const float tts_top_p = 0.95;
        static const float tts_temperature = 1.0;
        static const int tts_eos_token = 8192;
        for (auto &text : texts) {
            LOGI("[TTS] LLM input text: %s", text.c_str());
            auto text_tokens = tokenizer_encode(model_id, text);
            if (text_tokens.empty()) {
                LOGE("[TTS] Text tokenizer encode failed");
                generation_finished = true;
                return;
            }

            clear_state(model_id);
            Tensor1D logits;

            std::vector<int> input_tokens = {tts_tag_token_offset + 2}; // tag_2
            for (int i = 0; i < text_tokens.size(); i++) {
                input_tokens.push_back(text_tokens[i]);
            }
            input_tokens.push_back(tts_tag_token_offset + 0); // tag_0
            for (int i = 0; i < global_tokens.size(); i++) {
                input_tokens.push_back(global_tokens[i]); // global_tokens are already offset by global_token_offset
            }
            input_tokens.push_back(tts_tag_token_offset + 1); // tag_1

            int ret = eval_logits(model_id, input_tokens, logits);
            if (ret || logits.data_ptr == nullptr) {
                LOGE("[TTS] Error evaluating logits");
                generation_finished = true;
                return;
            }

            tensor1d_set_f32(logits, (size_t)tts_eos_token, -1e9f);

            for (int i = 0; i < tts_max_length; i++) {
                int idx = model->sampler->sample(logits, tts_tag_token_offset, tts_temperature, tts_top_k, tts_top_p);
                if (idx == tts_eos_token) {
                    LOGI("[TTS] EOS token found");
                    break;
                }

                output_tokens.push_back(idx);
                int ret = eval_logits(model_id, idx, logits);
                if (ret || logits.data_ptr == nullptr) {
                    LOGE("[TTS] Error evaluating logits");
                    generation_finished = true;
                    return;
                }
            }
        }
        generation_finished = true;
    });

    double ttfa = 0.0;
    std::thread detokenize_thread([&]() {
        tts_detokenize_thread_main(_sparktts.get(), this, global_tokens, output_tokens, generation_finished, ttfa, total_start);
    });

    llm_inference_thread.join();
    detokenize_thread.join();

    LOGI("[TTS] %s: LLM output tokens: %d", __func__, output_tokens.size());
    LOGI("[TTS] %s: LLM prefill speed: %f tokens/s", __func__, get_avg_prefill_speed(model_id));
    LOGI("[TTS] %s: LLM decode speed: %f tokens/s", __func__, get_avg_decode_speed(model_id));
    if (!_tts_output_samples_buffer.empty()) {
        save_samples_to_wav(_tts_output_samples_buffer, output_wav_path, 16000);
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
    LOGI("[TTS] %s: Total time: %lf ms", __func__, total_duration);
    LOGI("[TTS] %s: Output audio length: %lf s", __func__, _tts_output_samples_buffer.size() / 16000.0);
    LOGI("[TTS] %s: RTF: %lf", __func__, total_duration / 1e3f * 16000.0 / _tts_output_samples_buffer.size());
    LOGI("[TTS] %s: TTFA: %lf ms", __func__, ttfa);

    set_is_generating(model_id, false);
    return RWKV_SUCCESS;
}
#endif

int Runtime::clear_state(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);

    if (model->backend != nullptr) {
        model->backend->zero_state();
        int max_supported_batch_size = *std::max_element(model->backend->supported_batch_sizes.begin(), model->backend->supported_batch_sizes.end());
        if (max_supported_batch_size > 1) {
            for (int i = 0; i < max_supported_batch_size; i++) {
                model->backend->zero_state_on_batch_slot(i);
            }
        }
    }
    return RWKV_SUCCESS;
}

int Runtime::gen_completion_batch(int model_id, std::vector<std::string> prompts, int batch_size, int max_length, int stop_code, void (*callback_batch)(const int, const char **, const int*, const char **), bool disable_cache) {
    if (_models.find(model_id) == _models.end()) {
        LOGE("gen_completion_batch: Model ID %d not found", model_id);
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);
    if (model->backend == nullptr || model->tokenizer == nullptr) {
        LOGE("gen_completion_batch: Backend or tokenizer for model ID %d not found", model_id);
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    bool supported = false;
    for (auto size : model->backend->supported_batch_sizes) {
        if (batch_size == size) {
            supported = true;
            break;
        }
    }
    if (!supported) {
        LOGE("gen_completion_batch: batch size %d is not supported\n", batch_size);
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_UNSUPPORTED;
    }

    if (prompts.size() != batch_size) {
        LOGE("gen_completion_batch: prompts size %d is not equal to batch size %d\n", prompts.size(), batch_size);
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    model->is_generating = true;
    model->stop_signal = false;

    model->response_buffer_batch.resize(batch_size);
    model->response_buffer_ids_batch.resize(batch_size);
    model->response_buffer_decoded_tokens_batch.resize(batch_size);
    model->response_buffer_eos_found_batch.resize(batch_size);

    _clear_speed_samples(*model);

    std::vector<int> decoded_idx_batch(batch_size);
    std::vector<std::string> decoded_text_batch(batch_size);
    std::vector<state_node*> nodes_batch(batch_size);
    Tensor1D logits;

    std::vector<std::map<int, float>> occurences_batch(batch_size);
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        model->response_buffer_batch[batch_idx] = "";
        model->response_buffer_ids_batch[batch_idx].clear();
        model->response_buffer_decoded_tokens_batch[batch_idx] = 0;
        model->response_buffer_eos_found_batch[batch_idx] = false;

        std::vector<int> ids = model->tokenizer->encode(prompts[batch_idx]);
        std::vector<int> tokens_to_prefill;
        if (!disable_cache) {
            nodes_batch[batch_idx] = model->backend->match_and_load_state(ids, tokens_to_prefill);
        } else {
            nodes_batch[batch_idx] = model->backend->state_root.get();
            tokens_to_prefill = ids;
        }
        _prefill_progress_start(model_id, tokens_to_prefill.size());

        int checkpoint_interval = _get_prefill_checkpoint_interval((int) tokens_to_prefill.size());
        for (int j = 0; j < tokens_to_prefill.size(); j += checkpoint_interval) {
            std::vector<int> tokens_to_prefill_chunk = std::vector<int>(tokens_to_prefill.begin() + j, tokens_to_prefill.begin() + std::min(j + checkpoint_interval, (int)tokens_to_prefill.size()));
            int ret = eval_logits(model_id, tokens_to_prefill_chunk, logits);
            if (ret || logits.data_ptr == nullptr) {
                LOGE("gen_completion_batch: Error evaluating logits");
                model->is_generating = false;
                return ret;
            }
            if (!disable_cache) {
                ret = model->backend->register_state_checkpoint(nodes_batch[batch_idx], tokens_to_prefill_chunk, logits);
                if (ret) {
                    LOGE("gen_completion_batch: Error registering state checkpoint");
                    model->is_generating = false;
                    return ret;
                }
                LOGI("registered state for text: \"%s\"", escape_special_chars(model->tokenizer->decode(nodes_batch[batch_idx]->ids)).c_str());
            }
        }
        _prefill_progress_finish(model_id);

        model->response_buffer_batch[batch_idx] = prompts[batch_idx];
        model->response_buffer_ids_batch[batch_idx] = ids;
        model->response_buffer_decoded_tokens_batch[batch_idx] = (int)ids.size();

        if (logits.data_ptr == nullptr) {
            if (!nodes_batch[batch_idx]->logits.empty()) {
                logits = Tensor1D::make(nodes_batch[batch_idx]->logits.data(), TensorDType::F32, (size_t)model->backend->get_num_vocab()).copy();
            } else {
                LOGE("no logits found, neither from saved state nor from new tokens to prefill\n");
                model->is_generating = false;
                return RWKV_ERROR_RUNTIME;
            }
        }

        model->sampler->apply_penalties(logits, model->backend->get_num_vocab(), occurences_batch[batch_idx],
            model->sampler->get_token_banned(), model->sampler->get_presence_penalty(),
            model->sampler->get_frequency_penalty(), model->sampler->get_penalty_decay());
        decoded_idx_batch[batch_idx] = model->sampler->sample(logits, model->backend->get_num_vocab());
    }

    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        model->backend->set_state_on_batch_slot(batch_idx, nodes_batch[batch_idx]->state);
    }

    for (int i = 0; i < max_length; i++) {
        if (i != 0) {
            for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
                Tensor1D view = tensor1d_subview(logits, (size_t)batch_idx * (size_t)model->backend->get_num_vocab(), (size_t)model->backend->get_num_vocab());
                model->sampler->apply_penalties(view, model->backend->get_num_vocab(), occurences_batch[batch_idx],
                    model->sampler->get_token_banned(), model->sampler->get_presence_penalty(),
                    model->sampler->get_frequency_penalty(), model->sampler->get_penalty_decay());
            }
            decoded_idx_batch = model->sampler->sample_batch(logits, model->backend->get_num_vocab(), model->backend->get_num_vocab(), batch_size);
        }

        for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
            if (!model->response_buffer_eos_found_batch[batch_idx]) {
                model->response_buffer_eos_found_batch[batch_idx] = (decoded_idx_batch[batch_idx] == stop_code);
                if (model->response_buffer_eos_found_batch[batch_idx]) {
                    continue;
                }
                model->response_buffer_ids_batch[batch_idx].push_back(decoded_idx_batch[batch_idx]);
                if (callback_batch) {
                    decoded_text_batch[batch_idx] = model->tokenizer->decode(decoded_idx_batch[batch_idx]);
                    model->response_buffer_batch[batch_idx] += decoded_text_batch[batch_idx];
                    model->response_buffer_decoded_tokens_batch[batch_idx] = (int)model->response_buffer_ids_batch[batch_idx].size();
                }
            }
        }

        int ret = eval_logits_batch_decode(model_id, decoded_idx_batch, logits);
        if (ret) {
            model->is_generating = false;
            LOGE("failed to eval logits\n");
            return ret;
        }

        if (std::all_of(model->response_buffer_eos_found_batch.begin(), model->response_buffer_eos_found_batch.end(), [](bool eos_found) { return eos_found; }) || model->stop_signal) {
            break;
        }

        for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
            occurences_batch[batch_idx][decoded_idx_batch[batch_idx]]++;
        }
    }

    model->is_generating = false;
    model->stop_signal = false;
    return RWKV_SUCCESS;
}

int Runtime::gen_completion(int model_id, std::string prompt, int max_length, int stop_code, void (*callback)(const char *, const int, const char *), bool disable_cache) {
    if (_models.find(model_id) == _models.end()) {
        LOGE("gen_completion: Model ID %d not found", model_id);
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);
    if (model->backend == nullptr || model->tokenizer == nullptr) {
        LOGE("gen_completion: Backend or tokenizer for model ID %d not found", model_id);
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    model->response_buffer = "";
    model->response_buffer_ids.clear();
    model->response_buffer_decoded_tokens = 0;
    model->response_buffer_eos_found = false;
    model->is_generating = true;
    model->stop_signal = false;
    model->sampler->clear_occurences();

    _clear_speed_samples(*model);

    std::vector<int> ids = model->tokenizer->encode(prompt);
    std::vector<int> tokens_to_prefill;
    state_node* node = nullptr;
    if (!disable_cache) {
        node = model->backend->match_and_load_state(ids, tokens_to_prefill);
    } else {
        tokens_to_prefill = ids;
        node = model->backend->state_root.get();
    }
    _prefill_progress_start(model_id, tokens_to_prefill.size());

    Tensor1D logits;
    int checkpoint_interval = _get_prefill_checkpoint_interval((int) tokens_to_prefill.size());
    for (int j = 0; j < tokens_to_prefill.size(); j += checkpoint_interval) {
        std::vector<int> tokens_to_prefill_chunk = std::vector<int>(tokens_to_prefill.begin() + j, tokens_to_prefill.begin() + std::min(j + checkpoint_interval, (int)tokens_to_prefill.size()));
        int ret = eval_logits(model_id, tokens_to_prefill_chunk, logits);
        if (ret || logits.data_ptr == nullptr) {
            LOGE("gen_completion: Error evaluating logits");
            model->is_generating = false;
            return ret;
        }
        if (!disable_cache) {
            ret = model->backend->register_state_checkpoint(node, tokens_to_prefill_chunk, logits);
            if (ret) {
                LOGE("gen_completion: Error registering state checkpoint");
                model->is_generating = false;
                return ret;
            }
            LOGI("registered state for text: \"%s\"", escape_special_chars(model->tokenizer->decode(node->ids)).c_str());
        }
    }
    _prefill_progress_finish(model_id);

    model->response_buffer = prompt;
    model->response_buffer_ids = ids;
    model->response_buffer_decoded_tokens = (int)ids.size();
    static int idx = 0;
    if (logits.data_ptr == nullptr) {
        if (!node->logits.empty()) {
            logits = Tensor1D::make(node->logits.data(), TensorDType::F32, (size_t)model->backend->get_num_vocab()).copy();
        } else {
            LOGE("gen_completion: no logits available after prefill");
            model->is_generating = false;
            return RWKV_ERROR_RUNTIME;
        }
    }
    for (int i = 0; i < max_length; i++) {
        model->sampler->apply_penalties(logits, model->backend->get_num_vocab());
        idx = model->sampler->sample(logits, model->backend->get_num_vocab());

        model->response_buffer_eos_found = (idx == stop_code);
        model->response_buffer_ids.push_back(idx);
        int ret = eval_logits(model_id, idx, logits);
        if (ret) {
            model->is_generating = false;
            LOGE("failed to eval logits\n");
            return ret;
        }
        if (callback) {
            std::string next = model->tokenizer->decode(idx);
            model->response_buffer += next;
            model->response_buffer_decoded_tokens = (int)model->response_buffer_ids.size();
            callback(model->response_buffer.c_str(), idx, next.c_str());
        }

        if (model->response_buffer_eos_found || model->stop_signal) {
            break;
        }

        model->sampler->update_occurences(idx);
    }

    model->is_generating = false;
    model->stop_signal = false;
    return RWKV_SUCCESS;
}

int Runtime::gen_completion_singletoken_topk(int model_id, std::string prompt, int top_k, std::vector<std::string> &candidate_output_texts, void (*callback)(const char *, const int, const char *)) {
    if (_models.find(model_id) == _models.end()) {
        LOGE("gen_completion: Model ID %d not found", model_id);
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);
    if (model->backend == nullptr || model->tokenizer == nullptr) {
        LOGE("gen_completion: Backend or tokenizer for model ID %d not found", model_id);
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    _clear_speed_samples(*model);

    model->is_generating = true;

    std::vector<int> ids = model->tokenizer->encode(prompt);
    std::vector<int> tokens_to_prefill;
    state_node* node = model->backend->match_and_load_state(ids, tokens_to_prefill);
    _prefill_progress_start(model_id, tokens_to_prefill.size());

    Tensor1D logits;
    // The target usage requires more frequent state checkpoints.
    //
    // Save checkpoints at prompt-length fractions: 1/2, 3/4, 7/8, ... (i.e. 1 - 1/2^k).
    // If `match_and_load_state()` already loaded a cached prefix, we only prefill the suffix
    // and skip checkpoint boundaries that fall inside the cached prefix.
    const int total_prompt_tokens = (int)ids.size();
    const int cached_prefix_tokens = node ? (int)node->ids.size() : 0;
    const int total_to_prefill = (int)tokens_to_prefill.size();

    int j = 0;
    if (total_to_prefill > 0) {
        int last_end = 0;
        for (int k = 1; k < 32; k++) {
            int remaining = total_prompt_tokens >> k;               // floor(total / 2^k)
            int global_end = total_prompt_tokens - remaining;       // total * (1 - 1/2^k)
            if (global_end <= cached_prefix_tokens) {
                continue;
            }
            int end = global_end - cached_prefix_tokens;
            end = std::min(end, total_to_prefill);
            if (end <= last_end) {
                continue;
            }

            std::vector<int> tokens_to_prefill_chunk =
                std::vector<int>(tokens_to_prefill.begin() + j, tokens_to_prefill.begin() + end);

            j = end;
            last_end = end;

            int ret = eval_logits(model_id, tokens_to_prefill_chunk, logits);
            if (ret || logits.data_ptr == nullptr) {
                LOGE("gen_completion: Error evaluating logits");
                model->is_generating = false;
                return ret;
            }
            ret = model->backend->register_state_checkpoint(node, tokens_to_prefill_chunk, logits);
            if (ret) {
                LOGE("gen_completion: Error registering state checkpoint");
                model->is_generating = false;
                return ret;
            }
            LOGI("registered state for text: \"%s\"", escape_special_chars(model->tokenizer->decode(node->ids)).c_str());

            if (j >= total_to_prefill) {
                break;
            }
        }
    }

    // Safety fallback: if, for any reason, we didn't reach the end, finish prefill.
    for (; j < tokens_to_prefill.size();) {
        std::vector<int> tokens_to_prefill_chunk =
            std::vector<int>(tokens_to_prefill.begin() + j, tokens_to_prefill.end());
        j = (int)tokens_to_prefill.size();
        int ret = eval_logits(model_id, tokens_to_prefill_chunk, logits);
        if (ret || logits.data_ptr == nullptr) {
            LOGE("gen_completion: Error evaluating logits");
            model->is_generating = false;
            return ret;
        }
        ret = model->backend->register_state_checkpoint(node, tokens_to_prefill_chunk, logits);
        if (ret) {
            LOGE("gen_completion: Error registering state checkpoint");
            model->is_generating = false;
            return ret;
        }
        LOGI("registered state for text: \"%s\"", escape_special_chars(model->tokenizer->decode(node->ids)).c_str());
    }
    _prefill_progress_finish(model_id);

    static int idx = 0;
    if (logits.data_ptr == nullptr) {
        if (!node->logits.empty()) {
            logits = Tensor1D::make(node->logits.data(), TensorDType::F32, (size_t)model->backend->get_num_vocab()).copy();
        } else {
            LOGE("gen_completion: no logits available after prefill");
            model->is_generating = false;
            return RWKV_ERROR_RUNTIME;
        }
    }

    std::vector<int> top_k_indices = model->sampler->sample_topk_greedy(logits, model->backend->get_num_vocab(), top_k);

    candidate_output_texts.clear();
    for (int i = 0; i < top_k; i++) {
        candidate_output_texts.push_back(model->tokenizer->decode(top_k_indices[i]));
    }
    model->is_generating = false;
    model->stop_signal = false;
    return RWKV_SUCCESS;
}

int Runtime::run_evaluation(int model_id, std::string source_text, std::string target_text, bool &correct, float &logits_val, std::string &output_text, bool insert_bos_token) {
    if (_models.find(model_id) == _models.end()) {
        LOGE("run_evaluation: Model ID %d not found", model_id);
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);
    if (model->backend == nullptr || model->tokenizer == nullptr) {
        LOGE("run_evaluation: Backend or tokenizer for model ID %d not found", model_id);
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    static auto softmax_and_argmax = [](float *logits, size_t size) -> int {
        int max_idx = std::max_element(logits, logits + size) - logits;
        float max_val = logits[max_idx];
        float sum = 0;
        for (size_t i = 0; i < size; i++) {
            logits[i] = std::exp((logits[i] - max_val));
            sum += logits[i];
        }
        for (size_t i = 0; i < size; i++) {
            logits[i] /= sum;
        }
        return max_idx;
    };

    auto source_ids = model->tokenizer->encode(source_text);
    auto target_ids = model->tokenizer->encode(target_text);

    if (insert_bos_token) {
        source_ids.insert(source_ids.begin(), 0);
    }

    Tensor1D logits;
    clear_state(model_id);
    int ret = eval_logits(model_id, source_ids, logits);
    if (ret || logits.data_ptr == nullptr) {
        LOGE("run_evaluation: Error evaluating logits");
        return ret;
    }

    correct = true;
    logits_val = 0;
    const int vocab = model->backend->get_num_vocab();
    std::vector<float> logits_f32_copy((size_t)vocab);
    std::vector<int> output_ids;
    for (int i = 0; i < target_ids.size(); i++) {
        // Evaluation uses full softmax, so we always make a fp32 copy here.
        // NOTE: softmax_and_argmax modifies the buffer in-place.
        if (logits.data_ptr == nullptr || logits.count < (size_t)vocab) {
            LOGE("run_evaluation: invalid logits tensor");
            return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
        }
        if (logits.dtype == TensorDType::F32) {
            std::copy_n(reinterpret_cast<const float*>(logits.data_ptr), vocab, logits_f32_copy.data());
        } else if (logits.dtype == TensorDType::F16) {
            const half_float::half* h = reinterpret_cast<const half_float::half*>(logits.data_ptr);
            for (int j = 0; j < vocab; ++j) logits_f32_copy[j] = (float)h[j];
        } else {
            LOGE("run_evaluation: unsupported logits dtype");
            return RWKV_ERROR_UNSUPPORTED;
        }

        auto output_id = softmax_and_argmax(logits_f32_copy.data(), (size_t)vocab);
        output_ids.push_back(output_id);
        logits_val += std::log(logits_f32_copy[target_ids[i]]);
        if (output_id != target_ids[i]) {
            correct = false;
        }
        if (i != target_ids.size() - 1) {
            ret = eval_logits(model_id, target_ids[i], logits);
            if (ret || logits.data_ptr == nullptr) {
                LOGE("run_evaluation: Error evaluating logits");
                return ret;
            }
        }
    }

    output_text = model->tokenizer->decode(output_ids);

    return RWKV_SUCCESS;
}

double Runtime::get_avg_decode_speed(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return 0.0;
    }
    auto &model = _models.at(model_id);
    double speed_from_backend = model->backend->get_decode_speed();
    if (speed_from_backend > 0) {
        return speed_from_backend;
    }

    double speed = 0.0;
    {
        std::lock_guard<std::mutex> lock(model->speed_samples_mutex);
        speed = _compute_trimmed_mean_speed_tokens_per_s(model->decode_samples_us, _speed_trim_ratio_total);
    }
    if (speed > 0.0) {
        _decode_speed = speed;
        return speed;
    }
    return (_decode_speed < 0) ? 0.0 : _decode_speed;
}

double Runtime::get_avg_prefill_speed(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return 0.0;
    }
    auto &model = _models.at(model_id);
    double speed_from_backend = model->backend->get_prefill_speed();
    if (speed_from_backend > 0) {
        return speed_from_backend;
    }

    double speed = 0.0;
    {
        std::lock_guard<std::mutex> lock(model->speed_samples_mutex);
        speed = _compute_trimmed_mean_speed_tokens_per_s(model->prefill_samples_us, _speed_trim_ratio_total);
    }
    if (speed > 0.0) {
        _prefill_speed = speed;
        return speed;
    }
    return (_prefill_speed < 0) ? 0.0 : _prefill_speed;
}

void Runtime::reset_inference_speed_stats(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return;
    }

    auto &model = _models.at(model_id);
    _clear_speed_samples(*model);
    _decode_speed = -1;
    _prefill_speed = -1;
}

void Runtime::set_sampler_params(int model_id, float temperature, int top_k, float top_p) {
    if (_models.find(model_id) == _models.end()) {
        return;
    }
    auto &model = _models.at(model_id);
    model->sampler->set_temperature(temperature);
    model->sampler->set_top_k(top_k);
    model->sampler->set_top_p(top_p);
}

void Runtime::set_sampler_params_on_batch_slot(int model_id, int slot, float temperature, int top_k, float top_p) {
    if (_models.find(model_id) == _models.end()) {
        return;
    }
    auto &model = _models.at(model_id);
    model->sampler->set_temperature_on_batch_slot(slot, temperature);
    model->sampler->set_top_k_on_batch_slot(slot, top_k);
    model->sampler->set_top_p_on_batch_slot(slot, top_p);
}

void Runtime::set_penalty_params_on_batch_slot(int model_id, int slot, float presence_penalty, float frequency_penalty, float penalty_decay) {
    if (_models.find(model_id) == _models.end()) {
        return;
    }
    auto &model = _models.at(model_id);
    model->sampler->set_presence_penalty_on_batch_slot(slot, presence_penalty);
    model->sampler->set_frequency_penalty_on_batch_slot(slot, frequency_penalty);
    model->sampler->set_penalty_decay_on_batch_slot(slot, penalty_decay);
}

void Runtime::set_is_generating(int model_id, bool is_generating) {
    if (_models.find(model_id) == _models.end()) {
        return;
    }
    auto &model = _models.at(model_id);
    model->is_generating = is_generating;
}

void Runtime::set_stop_signal(int model_id, bool stop_signal) {
    if (_models.find(model_id) == _models.end()) {
        return;
    }
    auto &model = _models.at(model_id);
    model->stop_signal = stop_signal;
}

bool Runtime::get_stop_signal(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return false;
    }
    auto &model = _models.at(model_id);
    return model->stop_signal;
}

bool Runtime::is_generating(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return false;
    }
    auto &model = _models.at(model_id);
    return model->is_generating;
}

float Runtime::get_temperature(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return 1.0f;
    }
    auto &model = _models.at(model_id);
    return model->sampler->get_temperature();
}

float Runtime::get_temperature_on_batch_slot(int model_id, int slot) {
    if (_models.find(model_id) == _models.end()) {
        return 1.0f;
    }
    auto &model = _models.at(model_id);
    return model->sampler->get_temperature_on_batch_slot(slot);
}

int Runtime::get_top_k(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return 1;
    }
    auto &model = _models.at(model_id);
    return model->sampler->get_top_k();
}

int Runtime::get_top_k_on_batch_slot(int model_id, int slot) {
    if (_models.find(model_id) == _models.end()) {
        return 1;
    }
    auto &model = _models.at(model_id);
    return model->sampler->get_top_k_on_batch_slot(slot);
}

float Runtime::get_top_p(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return 1.0f;
    }
    auto &model = _models.at(model_id);
    return model->sampler->get_top_p();
}

float Runtime::get_top_p_on_batch_slot(int model_id, int slot) {
    if (_models.find(model_id) == _models.end()) {
        return 1.0f;
    }
    auto &model = _models.at(model_id);
    return model->sampler->get_top_p_on_batch_slot(slot);
}

void Runtime::set_penalty_params(int model_id, float presence_penalty, float frequency_penalty, float penalty_decay) {
    if (_models.find(model_id) == _models.end()) {
        return;
    }
    auto &model = _models.at(model_id);
    model->sampler->set_presence_penalty(presence_penalty);
    model->sampler->set_frequency_penalty(frequency_penalty);
    model->sampler->set_penalty_decay(penalty_decay);
}

float Runtime::get_presence_penalty(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return 0.0f;
    }
    auto &model = _models.at(model_id);
    return model->sampler->get_presence_penalty();
}

float Runtime::get_presence_penalty_on_batch_slot(int model_id, int slot) {
    if (_models.find(model_id) == _models.end()) {
        return 0.0f;
    }
    auto &model = _models.at(model_id);
    return model->sampler->get_presence_penalty_on_batch_slot(slot);
}

float Runtime::get_frequency_penalty_on_batch_slot(int model_id, int slot) {
    if (_models.find(model_id) == _models.end()) {
        return 0.0f;
    }
    auto &model = _models.at(model_id);
    return model->sampler->get_frequency_penalty_on_batch_slot(slot);
}

float Runtime::get_penalty_decay_on_batch_slot(int model_id, int slot) {
    if (_models.find(model_id) == _models.end()) {
        return 0.0f;
    }
    auto &model = _models.at(model_id);
    return model->sampler->get_penalty_decay_on_batch_slot(slot);
}

float Runtime::get_frequency_penalty(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return 0.0f;
    }
    auto &model = _models.at(model_id);
    return model->sampler->get_frequency_penalty();
}

float Runtime::get_penalty_decay(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return 0.0f;
    }
    auto &model = _models.at(model_id);
    return model->sampler->get_penalty_decay();
}

void Runtime::set_user_role(int model_id, std::string role) {
    if (_models.find(model_id) == _models.end()) {
        return;
    }
    auto &model = _models.at(model_id);
    model->user_role = role;
}

bool Runtime::get_space_after_roles(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return true;
    }
    auto &model = _models.at(model_id);
    return model->space_after_roles;
}

void Runtime::set_response_role(int model_id, std::string role) {
    if (_models.find(model_id) == _models.end()) {
        return;
    }
    auto &model = _models.at(model_id);
    model->response_role = role;
}

std::string Runtime::get_user_role(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return "";
    }
    return _models.at(model_id)->user_role;
}

std::string Runtime::get_response_role(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return "";
    }
    return _models.at(model_id)->response_role;
}

void Runtime::set_bos_token(int model_id, std::string token) {
    if (_models.find(model_id) == _models.end()) {
        return;
    }
    auto &model = _models.at(model_id);
    model->bos_token = token;
}

void Runtime::set_eos_token(int model_id, std::string token) {
    if (_models.find(model_id) == _models.end()) {
        return;
    }
    auto &model = _models.at(model_id);
    model->eos_token = token;
    if (token == "\n\n") {
        model->stop_token_seqs = {{261}, {28329, 11}, {28324, 11}, {28331, 11}, {5585}};
    } else if (token == "\n") {
        model->stop_token_seqs = {{11}, {28329}, {28324}, {28331}, {261}, {5585}};
    } else {
        model->stop_token_seqs.clear();
        model->stop_token_seqs.push_back(model->tokenizer->encode(token));
    }
}

std::string Runtime::get_thinking_token(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return "";
    }
    auto &model = _models.at(model_id);
    return model->thinking_token;
}

void Runtime::set_thinking_token(int model_id, std::string thinking_token) {
    if (_models.find(model_id) == _models.end()) {
        return;
    }
    auto &model = _models.at(model_id);
    model->thinking_token = thinking_token;
}

void Runtime::set_space_after_roles(int model_id, bool space_after_roles) {
    if (_models.find(model_id) == _models.end()) {
        return;
    }
    auto &model = _models.at(model_id);
    model->space_after_roles = space_after_roles;
}

std::vector<int> Runtime::tokenizer_encode(int model_id, std::string text) {
    if (_models.find(model_id) == _models.end()) {
        return {};
    }
    auto &model = _models.at(model_id);
    if (model->tokenizer == nullptr) {
        return {};
    }
    return model->tokenizer->encode(text);
}

std::string Runtime::tokenizer_decode(int model_id, std::vector<int> ids) {
    if (_models.find(model_id) == _models.end()) {
        return "";
    }
    auto &model = _models.at(model_id);
    if (model->tokenizer == nullptr) {
        return "";
    }
    return model->tokenizer->decode(ids);
}

std::string Runtime::tokenizer_decode(int model_id, int id) {
    if (_models.find(model_id) == _models.end()) {
        return "";
    }
    auto &model = _models.at(model_id);
    if (model->tokenizer == nullptr) {
        return "";
    }
    return model->tokenizer->decode(id);
}

void Runtime::clear_response_buffer(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return;
    }
    auto &model = _models.at(model_id);
    model->response_buffer.clear();
    model->response_buffer_ids.clear();
    model->response_buffer_decoded_tokens = 0;
    model->response_buffer_eos_found = false;
    for (int i = 0; i < model->response_buffer_batch.size(); i++) {
        model->response_buffer_batch[i].clear();
        model->response_buffer_ids_batch[i].clear();
        if (i < model->response_buffer_decoded_tokens_batch.size()) {
            model->response_buffer_decoded_tokens_batch[i] = 0;
        }
        model->response_buffer_eos_found_batch[i] = false;
    }
}

void Runtime::backend_set_extra_str(int model_id, std::string str) {
    if (_models.find(model_id) == _models.end()) {
        return;
    }
    auto &model = _models.at(model_id);
    if (model->backend == nullptr) {
        return;
    }
    model->backend->extra_str = str;
}

int Runtime::release() {
    _models.clear();
#ifdef ENABLE_LLAMACPP
    if (_embedding) {
        _embedding->release();
        _embedding = nullptr;
    }
#endif
#ifdef ENABLE_TTS
    if (_sparktts) {
        _sparktts = nullptr;
    }
#endif
    return RWKV_SUCCESS;
}

int Runtime::get_vocab_size(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return 0;
    }
    auto &model = _models.at(model_id);
    if (model->backend == nullptr) {
        return 0;
    }
    return model->backend->get_num_vocab();
}

std::string Runtime::get_response_buffer_content(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return "";
    }
    auto &model = _models.at(model_id);
    const int total = (int)model->response_buffer_ids.size();
    for (int i = model->response_buffer_decoded_tokens; i < total; i++) {
        model->response_buffer += model->tokenizer->decode(model->response_buffer_ids[i]);
    }
    model->response_buffer_decoded_tokens = total;
    return model->response_buffer;
}

const std::vector<int32_t> Runtime::get_response_buffer_ids(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return {};
    }
    auto &model = _models.at(model_id);
    return model->response_buffer_ids;
}

int Runtime::get_response_buffer_tokens_count(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return 0;
    }
    auto &model = _models.at(model_id);
    return (int)model->response_buffer_ids.size();
}

std::vector<int> Runtime::get_response_buffer_tokens_count_batch(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return {};
    }
    auto &model = _models.at(model_id);
    std::vector<int> counts;
    for (int i = 0; i < model->response_buffer_ids_batch.size(); i++) {
        counts.push_back(model->response_buffer_ids_batch[i].size());
    }
    return counts;
}

int Runtime::calculate_tokens_count_from_text(int model_id, std::string text) {
    if (_models.find(model_id) == _models.end()) {
        return 0;
    }
    auto &model = _models.at(model_id);
    if (model->tokenizer == nullptr || text.empty()) {
        return 0;
    }
    return (int)model->tokenizer->encode(text).size();
}

int Runtime::calculate_tokens_count_from_messages(int model_id, std::vector<std::string> inputs, std::vector<std::string> roles_map) {
    if (_models.find(model_id) == _models.end()) {
        return 0;
    }
    auto &model = _models.at(model_id);
    if (model->tokenizer == nullptr || inputs.empty() || inputs[0].empty()) {
        return 0;
    }
    std::string input_text = apply_chat_template(model_id, inputs, false, false, roles_map);
    return (int)model->tokenizer->encode(input_text).size();
}

bool Runtime::get_response_buffer_eos_found(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return false;
    }
    auto &model = _models.at(model_id);
    return model->response_buffer_eos_found;
}

std::vector<std::string> Runtime::get_response_buffer_content_batch(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return {};
    }
    auto &model = _models.at(model_id);
    if (model->response_buffer_decoded_tokens_batch.size() < model->response_buffer_ids_batch.size()) {
        model->response_buffer_decoded_tokens_batch.resize(model->response_buffer_ids_batch.size(), 0);
    }
    for (size_t i = 0; i < model->response_buffer_ids_batch.size(); i++) {
        const int total = (int)model->response_buffer_ids_batch[i].size();
        int &decoded = model->response_buffer_decoded_tokens_batch[i];
        for (int j = decoded; j < total; j++) {
            model->response_buffer_batch[i] += model->tokenizer->decode(model->response_buffer_ids_batch[i][j]);
        }
        decoded = total;
    }
    return model->response_buffer_batch;
}

std::vector<std::vector<int32_t>> Runtime::get_response_buffer_ids_batch(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return {};
    }
    auto &model = _models.at(model_id);
    return model->response_buffer_ids_batch;
}

std::vector<bool> Runtime::get_response_buffer_eos_found_batch(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return {};
    }
    auto &model = _models.at(model_id);
    return model->response_buffer_eos_found_batch;
}

double Runtime::get_prefill_progress(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return 0.0;
    }
    auto &model = _models.at(model_id);
    std::lock_guard<std::mutex> progress_lock(model->prefill_progress_mutex);

    if (model->current_prefill_total_tokens > 0 && model->prefill_estimated_total_us > 0) {
        const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - model->prefill_progress_started_at
        ).count();
        const double estimated_progress = std::min(
            0.97,
            std::max(
                model->prefill_progress,
                (double) elapsed_us / (double) model->prefill_estimated_total_us * 0.97
            )
        );
        model->prefill_progress = estimated_progress;
    }

    return model->prefill_progress;
}

void Runtime::set_token_banned(int model_id, std::vector<int> token_banned) {
    if (_models.find(model_id) == _models.end()) {
        return;
    }
    auto &model = _models.at(model_id);
    if (model->sampler == nullptr) {
        return;
    }
    model->sampler->set_token_banned(token_banned);
}

std::vector<int> Runtime::get_loaded_model_ids() {
    std::vector<int> model_ids;
    for (const auto& pair : _models) {
        model_ids.push_back(pair.first);
    }
    return model_ids;
}

std::map<int, std::map<std::string, std::string>> Runtime::get_loaded_models_info() {
    std::map<int, std::map<std::string, std::string>> models_info;

    for (const auto& pair : _models) {
        int model_id = pair.first;
        const auto& model = pair.second;

        std::map<std::string, std::string> model_info;
        model_info["model_path"] = model->model_path;
        model_info["backend_name"] = model->backend_name;
        model_info["tokenizer_path"] = model->tokenizer_path;
        model_info["user_role"] = model->user_role;
        model_info["response_role"] = model->response_role;
        model_info["bos_token"] = model->bos_token;
        model_info["eos_token"] = model->eos_token;
        model_info["thinking_token"] = model->thinking_token;
        model_info["is_generating"] = model->is_generating ? "true" : "false";
        model_info["vocab_size"] = model->backend ? std::to_string(model->backend->get_num_vocab()) : "0";

        models_info[model_id] = model_info;
    }

    return models_info;
}

std::string& Runtime::get_model_path_by_id(int model_id) {
    static std::string empty_string;
    if (_models.find(model_id) == _models.end()) {
        return empty_string;
    }
    auto &model = _models.at(model_id);
    return model->model_path;
}

int Runtime::set_seed(int model_id, int32_t seed) {
    if (_models.find(model_id) == _models.end()) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    auto &model = _models.at(model_id);
    if (model->sampler == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    model->sampler->set_seed(seed);
    return RWKV_SUCCESS;
}

int Runtime::get_seed(int model_id) {
    if (_models.find(model_id) == _models.end()) {
        return 0;
    }
    auto &model = _models.at(model_id);
    if (model->sampler == nullptr) {
        return 0;
    }
    return model->sampler->get_seed();
}
} // namespace rwkvmobile
