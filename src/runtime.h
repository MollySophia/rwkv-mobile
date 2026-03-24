#ifndef RUNTIME_H
#define RUNTIME_H

#include <string>
#include <map>
#include <memory>
#include <deque>
#include <functional>
#include <cstdlib>
#include <any>
#include <thread>
#include <mutex>
#include <atomic>
#include <cstdint>
#include <chrono>
#include "backend.h"
#include "tokenizer.h"
#include "sampler.h"
#include "soc_detect.h"

#include "logger.h"
#ifdef ENABLE_LLAMACPP
#include "embedding/rwkv_embedding.h"
#endif
#include "multimodal/multimodal_encoder.h"

#ifdef ENABLE_TTS
#include "sparktts.h"
#if !defined(_WIN32)
#include "kaldifst/csrc/text-normalizer.h"
#endif
#endif

namespace rwkvmobile {

class Runtime;

struct ModelInstance {
    ~ModelInstance() {
        LOGI("[ModelInstance] Release model instance");
#if defined(ENABLE_VISION) || defined(ENABLE_WHISPER)
        if (multimodal_encoder) {
            multimodal_encoder = nullptr;
        }
#endif
        if (sampler) {
            sampler = nullptr;
        }
        if (tokenizer) {
            tokenizer = nullptr;
        }
        if (backend) {
            backend = nullptr;
        }
    }
    std::string model_path;
    std::string backend_name;
    std::string tokenizer_path;
    std::unique_ptr<execution_provider, std::function<void(execution_provider*)>> backend;
    std::unique_ptr<tokenizer_base, std::function<void(tokenizer_base*)>> tokenizer;
    std::unique_ptr<NucleusSampler> sampler;
    // Add other per-model states here, e.g., chat templates, stop codes etc.
    std::string user_role = "User";
    std::string response_role = "Assistant";
    std::string system_role = "System";
    std::string bos_token = "";
    std::string eos_token = "\n\n";
    // std::vector<std::string> stop_codes = {"\n\n", "\nUser"};
    // "\n\n", "。\n\n"，"…\n\n", "，\n\n"
    std::vector<std::vector<int>> stop_token_seqs = {{261}, {28329, 11}, {28324, 11}, {28331, 11}, {5585}};
    std::string thinking_token = "<think";
    bool space_after_roles = true;

    // Response buffer
    std::string response_buffer;
    std::vector<int32_t> response_buffer_ids;
    int response_buffer_decoded_tokens = 0;
    bool response_buffer_eos_found = false;

    std::vector<std::string> response_buffer_batch;
    std::vector<std::vector<int32_t>> response_buffer_ids_batch;
    std::vector<int> response_buffer_decoded_tokens_batch;
    std::vector<bool> response_buffer_eos_found_batch;

    // Generation status
    std::string prompt;
    bool is_generating = false;
    bool stop_signal = false;

    struct SpeedSample {
        int tokens = 0;
        int64_t duration_us = 0;
    };
    mutable std::mutex speed_samples_mutex;
    std::deque<SpeedSample> decode_samples_us;
    std::deque<SpeedSample> prefill_samples_us;

    // Prefill progress
    mutable std::mutex prefill_progress_mutex;
    int current_prefill_total_tokens = -1;
    int current_prefill_finished_tokens = 0;
    double prefill_progress = 0.0;
    std::chrono::steady_clock::time_point prefill_progress_started_at;
    int64_t prefill_estimated_total_us = 0;

#if defined(ENABLE_VISION) || defined(ENABLE_WHISPER)
    std::unique_ptr<MultimodalEncoder> multimodal_encoder;
#endif
};

class Runtime {
public:
    Runtime() {
#ifdef __ANDROID__
        setenv("KMP_DUPLICATE_LIB_OK", "1", 1);
#endif
        _soc_detect.detect_platform();
    };

    ~Runtime() {
        release();
    };
    int load_model(std::string model_path, std::string backend_name, std::string tokenizer_path, void * extra);
    int release_model(int model_id);
    int release();

    int eval_logits(int model_id, int id, Tensor1D & logits);
    int eval_logits(int model_id, std::vector<int> ids, Tensor1D & logits);
    int eval_logits_with_embeddings(int model_id, const float *embeddings, int n_tokens, Tensor1D & logits);
    int eval_logits_batch_decode(int model_id, std::vector<int> ids, Tensor1D & logits);

    // with history
    int chat(int model_id, std::vector<std::string> inputs, const int max_length,
        void (*callback)(const char *, const int, const char *) = nullptr, bool enable_reasoning = false, bool force_reasoning = false,
        bool add_generation_prompt = true, int force_lang = 0, std::vector<std::string> roles_map = {}
    );
    int chat_batch(int model_id, std::vector<std::vector<std::string>> inputs, const int max_length, const int batch_size,
        void (*callback_batch)(const int, const char **, const int*, const char **) = nullptr, bool enable_reasoning = false, bool force_reasoning = false,
        bool add_generation_prompt = true, int force_lang = 0, std::vector<std::vector<std::string>> roles_map = {}
    );
    int gen_completion(int model_id, std::string prompt, int max_length, int stop_code, void (*callback)(const char *, const int, const char *), bool disable_cache=false);
    int gen_completion_batch(int model_id, std::vector<std::string> prompts, int batch_size, int max_length, int stop_code, void (*callback_batch)(const int, const char **, const int*, const char **), bool disable_cache=false);
    int gen_completion_singletoken_topk(int model_id, std::string prompt, int top_k, std::vector<std::string> &candidate_output_texts, void (*callback)(const char *, const int, const char *));

    int save_state_by_history(int model_id, std::vector<std::string> history, std::string state_path);
    int load_history_state_to_memory(int model_id, std::string state_path);

    int run_evaluation(int model_id, std::string source_text, std::string target_text, bool &correct, float &logits_val, std::string &output_text, bool insert_bos_token = true);

    std::vector<int> get_supported_batch_sizes(int model_id);

    int set_prompt(int model_id, std::string prompt);
    std::string get_prompt(int model_id);

    int prefill_to_cache(int model_id, std::string text);

    int load_initial_state(int model_id, std::string state_path);
    void unload_initial_state(int model_id, std::string state_path);

    std::string get_response_buffer_content(int model_id);
    const std::vector<int32_t> get_response_buffer_ids(int model_id);
    int get_response_buffer_tokens_count(int model_id);
    std::vector<int> get_response_buffer_tokens_count_batch(int model_id);
    void clear_response_buffer(int model_id);
    bool get_response_buffer_eos_found(int model_id);

    int calculate_tokens_count_from_text(int model_id, std::string text);
    int calculate_tokens_count_from_messages(int model_id, std::vector<std::string> inputs, std::vector<std::string> roles_map = {});

    std::vector<std::string> get_response_buffer_content_batch(int model_id);
    std::vector<std::vector<int32_t>> get_response_buffer_ids_batch(int model_id);
    void clear_response_buffer_batch(int model_id);
    std::vector<bool> get_response_buffer_eos_found_batch(int model_id);
#ifdef ENABLE_VISION
    int load_vision_encoder(int model_id, std::string model_path, std::string adapter_path = "");
    int release_vision_encoder(int model_id);
#endif

    int set_image_unique_identifier(std::string unique_identifier);

    std::string _image_unique_identifier;

#ifdef ENABLE_WHISPER
    int load_whisper_encoder(int model_id, std::string model_path);
    int release_whisper_encoder(int model_id);
    int set_audio_prompt(int model_id, std::string path);
#endif

#ifdef ENABLE_TTS
    int sparktts_load_models(
        std::string wav2vec2_path,
        std::string bicodec_tokenizer_path,
        std::string bicodec_detokenizer_path
    );

    int sparktts_release_models();

    int run_spark_tts_zeroshot(int model_id, std::string tts_text, std::string prompt_audio_text, std::string prompt_audio_path, std::string output_wav_path);
    int run_spark_tts_with_properties(int model_id, std::string tts_text, std::string output_wav_path,
        std::string age, std::string gender, std::string emotion, std::string pitch, std::string speed);
    int run_spark_tts_with_global_tokens(int model_id, std::string tts_text, std::string output_wav_path, std::vector<int> global_tokens);

    int run_spark_tts_zeroshot_streaming(int model_id, std::string tts_text, std::string prompt_audio_text, std::string prompt_audio_path, std::string output_wav_path);
    int run_spark_tts_with_properties_streaming(int model_id, std::string tts_text, std::string output_wav_path,
        std::string age, std::string gender, std::string emotion, std::string pitch, std::string speed);
    int run_spark_tts_with_global_tokens_streaming(int model_id, std::string tts_text, std::string output_wav_path, std::vector<int> global_tokens);

    std::vector<int> _global_tokens_output;
    std::vector<int>& tts_get_global_tokens_output() {
        return _global_tokens_output;
    }

    std::vector<float>& tts_get_streaming_buffer() {
        return _tts_output_samples_buffer;
    }

    void tts_append_samples_to_buffer(std::vector<float>::iterator samples_begin, std::vector<float>::iterator samples_end) {
        std::lock_guard<std::mutex> lock(_tts_streaming_buffer_mutex);
        _tts_output_samples_buffer.insert(_tts_output_samples_buffer.end(), samples_begin, samples_end);
    }

    void tts_clear_streaming_buffer() {
        std::lock_guard<std::mutex> lock(_tts_streaming_buffer_mutex);
        _tts_output_samples_buffer.clear();
    }

    std::mutex _tts_streaming_buffer_mutex;

    int tts_register_text_normalizer(std::string path) {
#if !defined(_WIN32)
        if (!std::ifstream(path).good()) {
            LOGE("[TTS] Failed to load text normalizer file %s\n", path.c_str());
            return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
        }
        _tn_list.push_back(std::make_unique<kaldifst::TextNormalizer>(path));
        LOGI("[TTS] Loaded text normalizer file %s\n", path.c_str());
#endif
        return RWKV_SUCCESS;
    }

    int tts_clear_text_normalizer() {
#if !defined(_WIN32)
        _tn_list.clear();
#endif
        return RWKV_SUCCESS;
    }
#endif

    // for state management
    int clear_state(int model_id);

    // sampler and seed
    int set_seed(int model_id, int32_t seed);
    int get_seed(int model_id);

    void set_user_role(int model_id, std::string role);
    void set_response_role(int model_id, std::string role);
    void set_bos_token(int model_id, std::string token);
    void set_eos_token(int model_id, std::string token);
    void set_space_after_roles(int model_id, bool space_after_roles);
    std::string get_user_role(int model_id);
    std::string get_response_role(int model_id);
    std::string get_bos_token(int model_id);
    std::string get_eos_token(int model_id);
    bool get_space_after_roles(int model_id);

    std::string apply_chat_template(int model_id, std::vector<std::string> inputs, bool enable_reasoning = false,
        bool add_generation_prompt = true, std::vector<std::string> roles_map = {}
    );

    struct TokenChunk {
        std::vector<int> tokens;
        bool is_image;
        std::string image_path; // only valid when is_image is true
    };
    std::vector<TokenChunk> split_text_by_image_and_token_num(const std::string text, int max_tokens_per_chunk, int model_id);

    int get_vocab_size(int model_id);

    std::string get_thinking_token(int model_id);
    void set_thinking_token(int model_id, std::string thinking_token);

    void set_sampler_params(int model_id, float temperature, int top_k, float top_p);
    void set_penalty_params(int model_id, float presence_penalty, float frequency_penalty, float penalty_decay);
    void set_sampler_params_on_batch_slot(int model_id, int slot, float temperature, int top_k, float top_p);
    void set_penalty_params_on_batch_slot(int model_id, int slot, float presence_penalty, float frequency_penalty, float penalty_decay);

    float get_temperature(int model_id);
    int get_top_k(int model_id);
    float get_top_p(int model_id);
    float get_presence_penalty(int model_id);
    float get_frequency_penalty(int model_id);
    float get_penalty_decay(int model_id);

    float get_temperature_on_batch_slot(int model_id, int slot);
    int get_top_k_on_batch_slot(int model_id, int slot);
    float get_top_p_on_batch_slot(int model_id, int slot);
    float get_presence_penalty_on_batch_slot(int model_id, int slot);
    float get_frequency_penalty_on_batch_slot(int model_id, int slot);
    float get_penalty_decay_on_batch_slot(int model_id, int slot);

    void set_token_banned(int model_id, std::vector<int> token_banned);

    bool is_generating(int model_id);
    void set_is_generating(int model_id, bool is_generating);

    bool get_stop_signal(int model_id);
    void set_stop_signal(int model_id, bool stop_signal);

    std::string get_available_backends_str();
    int get_available_backend_ids(std::vector<int> &backend_ids);

    double get_avg_decode_speed(int model_id);
    double get_avg_prefill_speed(int model_id);
    double get_prefill_progress(int model_id);
    void reset_inference_speed_stats(int model_id);

    std::string get_state_cache_info(int model_id);

#ifdef ENABLE_LLAMACPP
    int load_embedding_model(std::string model_path) {
        if (_embedding == nullptr) {
            _embedding = std::make_unique<rwkv_embedding>();
        }
        return _embedding->load_model(model_path);
    }

    int release_embedding_model() {
        if (_embedding == nullptr) {
            return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
        }
        _embedding->release();
        _embedding = nullptr;
        return RWKV_SUCCESS;
    }

    int load_rerank_model(std::string model_path) {
        if (_embedding == nullptr) {
            _embedding = std::make_unique<rwkv_embedding>();
        }
        return _embedding->load_rerank_model(model_path);
    }

    std::vector<std::vector<float>> get_embedding(std::vector<std::string> inputs)  {
        if (_embedding == nullptr) {
            LOGE("Embedding model not loaded\n");
            return {};
        }
        return _embedding->get_embeddings(inputs);
    }

    std::vector<float> rerank(std::string query, const std::vector<std::string> &chunks)  {
        if (_embedding == nullptr) {
            LOGE("Embedding model not loaded\n");
            return {};
        }
        return _embedding->rerank(query, chunks);
    }
#endif

    // platform info
    const char * get_platform_name() {
        auto platform_name = _soc_detect.get_platform_name();
        LOGD("Platform name: %s", platform_name);
        return platform_name;
    }

    const char * get_soc_name() {
        auto soc_name = _soc_detect.get_soc_name();
        LOGD("SOC name: %s", soc_name);
        return soc_name;
    }

    const char * get_soc_partname() {
        auto soc_partname = _soc_detect.get_soc_partname();
        LOGD("SOC partname: %s", soc_partname);
        return soc_partname;
    }

    // backend
    std::string backend_id_to_str(int backend_id) {
        return backend_enum_to_str(backend_id);
    }
    int backend_str_to_id(std::string backend_str) {
        return backend_str_to_enum(backend_str);
    }

    void backend_set_extra_str(int model_id, std::string str);

    // tokenizer
    std::vector<int> tokenizer_encode(int model_id, std::string text);

    std::string tokenizer_decode(int model_id, std::vector<int> ids);

    std::string tokenizer_decode(int model_id, int id);

    // sampler
    int sampler_sample(int model_id, std::vector<float> logits);

    // get loaded models info
    std::vector<int> get_loaded_model_ids();
    std::map<int, std::map<std::string, std::string>> get_loaded_models_info();
    std::string& get_model_path_by_id(int model_id);

    // async load model status (for rwkvmobile_runtime_load_model_async)
    bool is_loading_model() const { return _load_model_in_progress.load(); }
    void start_load_model_async();
    void set_load_model_result(int result_code, int model_id);
    void get_load_model_result(int& result_code, int& model_id) const;
    float get_load_model_progress() const;

    // misc
    inline void set_cache_dir(std::string cache_dir) { _cache_dir = cache_dir; }

private:
    std::map<int, std::unique_ptr<ModelInstance>> _models;

    // async load model state
    std::atomic<bool> _load_model_in_progress{false};
    mutable std::mutex _load_model_result_mutex;
    int _load_model_result_code = 0;
    int _load_model_result_id = -1;
    execution_provider* _loading_backend = nullptr;
    mutable std::mutex _loading_backend_mutex;
    mutable std::mutex _load_progress_fallback_mutex;
    mutable float _load_progress_fallback = 0.f;
    mutable float _load_progress_fallback_step = 0.1f;

#ifdef ENABLE_LLAMACPP
    std::unique_ptr<rwkv_embedding> _embedding;
#endif

    double _prefill_speed = -1;
    double _decode_speed = -1;

    static constexpr size_t _speed_samples_max = 256; // sliding window size
    static constexpr double _speed_trim_ratio_total = 0.10; // keep 90%

    void _record_speed_sample(ModelInstance& model, bool is_prefill, int tokens, int64_t duration_us);
    static double _compute_trimmed_mean_speed_tokens_per_s(
        const std::deque<ModelInstance::SpeedSample>& samples,
        double trim_ratio_total
    );
    void _clear_speed_samples(ModelInstance& model);
    int _get_prefill_checkpoint_interval(int total_tokens) const;

    const int _prefill_chunk_size = 2048;

    void _prefill_progress_start(int model_id, int total_tokens);
    void _prefill_progress_finish(int model_id);

    std::string _cache_dir = "";

    soc_detect _soc_detect;

#ifdef ENABLE_TTS
    std::unique_ptr<sparktts> _sparktts;
#if !defined(_WIN32)
    std::vector<std::unique_ptr<kaldifst::TextNormalizer>> _tn_list;
#endif

    std::vector<float> _tts_output_samples_buffer;
#endif
};

} // namespace rwkvmobile

#endif
