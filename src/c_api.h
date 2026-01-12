#ifndef C_API_H
#define C_API_H

typedef void * rwkvmobile_runtime_t;
typedef void * rwkvmobile_server_t;

struct sampler_params {
    float temperature;
    int top_k;
    float top_p;
};

struct penalty_params {
    float presence_penalty;
    float frequency_penalty;
    float penalty_decay;
};

struct token_ids {
    int * ids;
    int len;
};

struct rwkvmobile_server_config {
    const char * host;
    int port;
    int threads;
    const char * model_name;
    int default_max_tokens;
    float temperature;
    int top_k;
    float top_p;
    float presence_penalty;
    float frequency_penalty;
    float penalty_decay;
    int has_temperature;
    int has_top_k;
    int has_top_p;
    int has_presence_penalty;
    int has_frequency_penalty;
    int has_penalty_decay;
};

struct response_buffer {
    char * content;
    int length;
    int eos_found;
};

struct response_buffer_batch {
    char ** contents;
    int * lengths;
    int * eos_founds;
    int batch_size;
};

struct tts_streaming_buffer {
    float * samples;
    int length;
};

struct supported_batch_sizes {
    int * sizes;
    int length;
};

struct model_info {
    int model_id;
    char * model_path;
    char * backend_name;
    char * tokenizer_path;
    char * user_role;
    char * response_role;
    char * bos_token;
    char * eos_token;
    char * thinking_token;
    int is_generating;
    int vocab_size;
};

struct loaded_models_list {
    struct model_info * models;
    int count;
};

struct evaluation_results {
    int * corrects;
    float * logits_vals;
    int count;
    char ** output_texts;
};

struct web_rwkv_args {
    int quant_type;    // 0: fp, 1: int8, 2: nf4
    int quant_layers;
};

struct batch_tokens_count {
    int * counts;
    int batch_size;
};

#ifdef __cplusplus
extern "C" {
#endif

enum {
    FORCE_LANG_NONE = 0,
    FORCE_LANG_CHN = 1,
};

int rwkvmobile_runtime_get_available_backend_names(char * backend_names_buffer, int buffer_size);

rwkvmobile_runtime_t rwkvmobile_runtime_init();

int rwkvmobile_runtime_release(rwkvmobile_runtime_t runtime);

struct rwkvmobile_server_config rwkvmobile_server_config_default();

rwkvmobile_server_t rwkvmobile_server_start(rwkvmobile_runtime_t runtime, int model_id, const struct rwkvmobile_server_config * config);

int rwkvmobile_server_stop(rwkvmobile_server_t server);

int rwkvmobile_server_wait(rwkvmobile_server_t server);

int rwkvmobile_server_release(rwkvmobile_server_t server);

int rwkvmobile_runtime_load_model(rwkvmobile_runtime_t runtime, const char * model_path, const char * backend_name, const char * tokenizer_path);

int rwkvmobile_runtime_load_model_with_extra(rwkvmobile_runtime_t runtime, const char * model_path, const char * backend_name, const char * tokenizer_path, void * extra);

int rwkvmobile_runtime_load_model_async(rwkvmobile_runtime_t runtime, const char * model_path, const char * backend_name, const char * tokenizer_path);

int rwkvmobile_runtime_load_model_with_extra_async(rwkvmobile_runtime_t runtime, const char * model_path, const char * backend_name, const char * tokenizer_path, void * extra);

int rwkvmobile_runtime_is_loading_model(rwkvmobile_runtime_t runtime);

void rwkvmobile_runtime_get_load_model_status(rwkvmobile_runtime_t runtime, int * result_code, int * model_id);

float rwkvmobile_runtime_get_load_model_progress(rwkvmobile_runtime_t runtime);

int rwkvmobile_runtime_release_model(rwkvmobile_runtime_t runtime, int model_id);

int rwkvmobile_runtime_eval_logits(rwkvmobile_runtime_t runtime, int model_id, const int *ids, int ids_len, float * logits, int logits_len);

int rwkvmobile_runtime_eval_chat_with_history_async(rwkvmobile_runtime_t handle, int model_id, const char ** inputs,
    const int num_inputs, const int max_tokens, void (*callback)(const char *, const int, const char *),
    int enable_reasoning, int force_reasoning, int force_lang, int add_generation_prompt
);

int rwkvmobile_runtime_stop_generation(rwkvmobile_runtime_t runtime, int model_id);

int rwkvmobile_runtime_is_generating(rwkvmobile_runtime_t runtime, int model_id);

int rwkvmobile_runtime_set_prompt(rwkvmobile_runtime_t runtime, int model_id, const char * prompt);

int rwkvmobile_runtime_get_prompt(rwkvmobile_runtime_t runtime, int model_id, char * prompt, const int buf_len);

int rwkvmobile_runtime_gen_completion_async(rwkvmobile_runtime_t runtime, int model_id, const char * prompt, const int max_tokens, const int stop_code, void (*callback)(const char *, const int, const char *), int disable_cache);

int rwkvmobile_runtime_gen_completion_batch_async(rwkvmobile_runtime_t runtime, int model_id, const char ** prompts, const int batch_size, const int max_tokens, const int stop_code, void (*callback_batch)(const int, const char **, const int*, const char **), int disable_cache);

int rwkvmobile_runtime_eval_chat_batch_with_history_async(rwkvmobile_runtime_t handle, int model_id, const char *** inputs, const int * num_inputs,
    const int batch_size, const int max_tokens, void (*callback_batch)(const int, const char **, const int*, const char **),
    int enable_reasoning, int force_reasoning, int force_lang, int add_generation_prompt
);

struct supported_batch_sizes rwkvmobile_runtime_get_supported_batch_sizes(rwkvmobile_runtime_t runtime, int model_id);

void rwkvmobile_runtime_free_supported_batch_sizes(struct supported_batch_sizes sizes);

int rwkvmobile_runtime_gen_completion(rwkvmobile_runtime_t runtime, int model_id, const char * prompt, const int max_tokens, const int stop_code, void (*callback)(const char *, const int, const char *), int disable_cache);

const char ** rwkvmobile_runtime_gen_completion_singletoken_topk(rwkvmobile_runtime_t handle, int model_id, const char * prompt, const int top_k);

int rwkvmobile_runtime_clear_state(rwkvmobile_runtime_t runtime, int model_id);

int rwkvmobile_runtime_load_initial_state(rwkvmobile_runtime_t runtime, int model_id, const char * state_path);

void rwkvmobile_runtime_unload_initial_state(rwkvmobile_runtime_t runtime, int model_id, const char * state_path);

int rwkvmobile_runtime_save_history_to_state(rwkvmobile_runtime_t runtime, int model_id, const char ** history, const int num_history, const char * state_path);

int rwkvmobile_runtime_load_history_state_to_memory(rwkvmobile_runtime_t runtime, int model_id, const char * state_path);

struct sampler_params rwkvmobile_runtime_get_sampler_params(rwkvmobile_runtime_t runtime, int model_id);

struct sampler_params rwkvmobile_runtime_get_sampler_params_on_batch_slot(rwkvmobile_runtime_t runtime, int model_id, int slot);

void rwkvmobile_runtime_set_sampler_params(rwkvmobile_runtime_t runtime, int model_id, struct sampler_params params);

void rwkvmobile_runtime_set_sampler_params_on_batch_slot(rwkvmobile_runtime_t runtime, int model_id, int slot, struct sampler_params params);

int rwkvmobile_runtime_set_seed(rwkvmobile_runtime_t runtime, int model_id, int seed);

int rwkvmobile_runtime_get_seed(rwkvmobile_runtime_t runtime, int model_id);

struct penalty_params rwkvmobile_runtime_get_penalty_params(rwkvmobile_runtime_t runtime, int model_id);

struct penalty_params rwkvmobile_runtime_get_penalty_params_on_batch_slot(rwkvmobile_runtime_t runtime, int model_id, int slot);

void rwkvmobile_runtime_set_penalty_params(rwkvmobile_runtime_t runtime, int model_id, struct penalty_params params);

void rwkvmobile_runtime_set_penalty_params_on_batch_slot(rwkvmobile_runtime_t runtime, int model_id, int slot, struct penalty_params params);

void rwkvmobile_runtime_add_adsp_library_path(const char * path);

void rwkvmobile_runtime_set_qnn_library_path(rwkvmobile_runtime_t runtime, const char * path);

double rwkvmobile_runtime_get_avg_decode_speed(rwkvmobile_runtime_t runtime, int model_id);

double rwkvmobile_runtime_get_avg_prefill_speed(rwkvmobile_runtime_t runtime, int model_id);

struct evaluation_results rwkvmobile_runtime_run_evaluation(rwkvmobile_runtime_t runtime, int model_id, const char * source_text, const char * target_text);

void rwkvmobile_runtime_free_evaluation_results(struct evaluation_results results);

// Vision
int rwkvmobile_runtime_load_vision_encoder(rwkvmobile_runtime_t runtime, int model_id, const char * encoder_path);

int rwkvmobile_runtime_load_vision_encoder_and_adapter(rwkvmobile_runtime_t runtime, int model_id, const char * encoder_path, const char * adapter_path);

int rwkvmobile_runtime_release_vision_encoder(rwkvmobile_runtime_t runtime, int model_id);

int rwkvmobile_runtime_set_image_unique_identifier(rwkvmobile_runtime_t runtime, const char * unique_identifier);

// Whisper
int rwkvmobile_runtime_load_whisper_encoder(rwkvmobile_runtime_t runtime, int model_id, const char * encoder_path);

int rwkvmobile_runtime_release_whisper_encoder(rwkvmobile_runtime_t runtime, int model_id);

int rwkvmobile_runtime_set_audio_prompt(rwkvmobile_runtime_t runtime, int model_id, const char * audio_path);

int rwkvmobile_runtime_set_token_banned(rwkvmobile_runtime_t runtime, int model_id, const int * token_banned, int token_banned_len);

int rwkvmobile_runtime_set_eos_token(rwkvmobile_runtime_t runtime, int model_id, const char * eos_token);

int rwkvmobile_runtime_set_bos_token(rwkvmobile_runtime_t runtime, int model_id, const char * bos_token);

int rwkvmobile_runtime_set_user_role(rwkvmobile_runtime_t runtime, int model_id, const char * user_role);

int rwkvmobile_runtime_set_space_after_roles(rwkvmobile_runtime_t runtime, int model_id, int space_after_roles);

int rwkvmobile_runtime_set_response_role(rwkvmobile_runtime_t runtime, int model_id, const char * response_role);

int rwkvmobile_runtime_set_thinking_token(rwkvmobile_runtime_t runtime, int model_id, const char * thinking_token);

struct response_buffer rwkvmobile_runtime_get_response_buffer_content(rwkvmobile_runtime_t runtime, int model_id);

struct response_buffer_batch rwkvmobile_runtime_get_response_buffer_content_batch(rwkvmobile_runtime_t runtime, int model_id);

void rwkvmobile_runtime_free_response_buffer(struct response_buffer buffer);

void rwkvmobile_runtime_free_response_buffer_batch(struct response_buffer_batch buffer);

struct token_ids rwkvmobile_runtime_get_response_buffer_ids(rwkvmobile_runtime_t runtime, int model_id);

int rwkvmobile_runtime_get_response_buffer_tokens_count(rwkvmobile_runtime_t runtime, int model_id);

struct batch_tokens_count rwkvmobile_runtime_get_response_buffer_tokens_count_batch(rwkvmobile_runtime_t runtime, int model_id);

int rwkvmobile_runtime_calculate_tokens_count_from_messages(rwkvmobile_runtime_t runtime, int model_id, const char ** inputs, const int num_inputs);

int rwkvmobile_runtime_calculate_tokens_count_from_text(rwkvmobile_runtime_t runtime, int model_id, const char * text);

void rwkvmobile_runtime_free_token_ids(struct token_ids ids);

// sparktts
int rwkvmobile_runtime_sparktts_load_models(rwkvmobile_runtime_t runtime, const char * wav2vec2_path, const char * bicodec_tokenizer_path, const char * bicodec_detokenizer_path);

int rwkvmobile_runtime_sparktts_release_models(rwkvmobile_runtime_t runtime);

int rwkvmobile_runtime_run_spark_tts_streaming_async(rwkvmobile_runtime_t runtime, int model_id, const char * tts_text, const char * prompt_audio_text, const char * prompt_audio_path, const char * output_wav_path);

int rwkvmobile_runtime_run_spark_tts_with_global_tokens_streaming_async(rwkvmobile_runtime_t runtime, int model_id, const char * tts_text, const char * output_wav_path, const int * global_tokens);

int rwkvmobile_runtime_run_spark_tts_with_properties_streaming_async(rwkvmobile_runtime_t runtime, int model_id, const char * tts_text, const char * output_wav_path, const char * age, const char * gender, const char * emotion, const char * pitch, const char * speed);

struct tts_streaming_buffer rwkvmobile_runtime_get_tts_streaming_buffer(rwkvmobile_runtime_t runtime);

int rwkvmobile_runtime_get_tts_streaming_buffer_length(rwkvmobile_runtime_t runtime);

void rwkvmobile_runtime_free_tts_streaming_buffer(struct tts_streaming_buffer buffer);

const int * rwkvmobile_runtime_get_tts_global_tokens_output(rwkvmobile_runtime_t runtime);

int rwkvmobile_runtime_tts_register_text_normalizer(rwkvmobile_runtime_t runtime, const char * path);

float rwkvmobile_runtime_get_prefill_progress(rwkvmobile_runtime_t runtime, int model_id);

int rwkvmobile_load_embedding_model(rwkvmobile_runtime_t runtime, const char *model_path);

int rwkvmobile_load_rerank_model(rwkvmobile_runtime_t runtime, const char *model_path);

int rwkvmobile_get_embedding(rwkvmobile_runtime_t runtime, const char **input, const int input_length,float **embedding);

// platform info
const char * rwkvmobile_get_platform_name();

const char * rwkvmobile_get_soc_name();

const char * rwkvmobile_get_soc_partname();

const char * rwkvmobile_get_htp_arch();

// logging

enum {
    RWKV_LOG_LEVEL_DEBUG = 0,
    RWKV_LOG_LEVEL_INFO,
    RWKV_LOG_LEVEL_WARN,
    RWKV_LOG_LEVEL_ERROR,
};

const char * rwkvmobile_dump_log();

const char * rwkvmobile_get_state_cache_info(rwkvmobile_runtime_t runtime, int model_id);

void rwkvmobile_free_state_cache_info(const char * state_cache_info);

void rwkvmobile_set_loglevel(int loglevel);

void rwkvmobile_set_cache_dir(rwkvmobile_runtime_t runtime, const char * cache_dir);

// get loaded models info
int rwkvmobile_runtime_get_loaded_model_ids(rwkvmobile_runtime_t runtime, int * model_ids, int max_count);

struct loaded_models_list rwkvmobile_runtime_get_loaded_models_info(rwkvmobile_runtime_t runtime);

void rwkvmobile_runtime_free_loaded_models_list(struct loaded_models_list list);

const char * rwkvmobile_runtime_get_model_path_by_id(rwkvmobile_runtime_t runtime, int model_id);

// conversion
int rwkvmobile_convert_pth_to_safetensors(const char * pth_path, const char * st_path);
#ifdef __cplusplus
}
#endif

#endif // C_API_H