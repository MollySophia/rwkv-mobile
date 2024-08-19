#ifndef C_API_H
#define C_API_H

typedef void * rwkvmobile_runtime_t;

#ifdef __cplusplus
extern "C" {
#endif

// ============================
// init runtime with backend name
// returns: runtime handle
rwkvmobile_runtime_t rwkvmobile_runtime_init_with_name(const char * backend_name);

// ============================
// load model file
// args: runtime handle, model file path
// returns: Error codes
int rwkvmobile_runtime_load_model(rwkvmobile_runtime_t runtime, const char * model_path);

// ============================
// load tokenizer from vocab_file
// args: runtime handle, vocab_file path
// returns: Error codes
int rwkvmobile_runtime_load_tokenizer(rwkvmobile_runtime_t runtime, const char * vocab_file);

// ============================
// eval logits with token id
// args: runtime handle, token id, buffer for logits output, logits buffer length
// note: buffer ptr must not be null; logits_len must be equal to the model's vocab_size
// returns: Error codes
int rwkvmobile_runtime_eval_logits(rwkvmobile_runtime_t runtime, const int *ids, int ids_len, float * logits, int logits_len);

// ============================
// get chat response message
// args: runtime handle, user role, response role, user input, char buffer for response output, response length limit
// note: response buffer should be allocated by the caller, and should be large enough to hold the response
// ============================
// example: rwkvmobile_runtime_eval_chat(runtime, "User", "Assistant", "hello", response, 1024);
// the text being passed to the model would be:
// "User: hello\n\nAssistant:"
// ============================
// response will stop when these conditions are met:
// 1. the model generates an end-of-sequence token (id = 0)
// 2. the model generates double-newline ("\n\n")
// 3. the response length reaches the limit
// ============================
// returns: Error codes
int rwkvmobile_runtime_eval_chat(rwkvmobile_runtime_t runtime, const char * user_role, const char * response_role, const char * user_input, char * response, const int max_length);

// ============================
// generate completion from prompt
// args: runtime handle, prompt text, char buffer for completion output, completion length
// note: completion buffer should be allocated by the caller, and should be large enough to hold the completion
// ============================
// response will stop when these conditions are met:
// 1. the model generates an end-of-sequence token (id = 0)
// 2. the model generates double-newline ("\n\n")
// 3. the response length reaches the limit
// ============================
// returns: Error codes
int rwkvmobile_runtime_gen_completion(rwkvmobile_runtime_t runtime, const char * prompt, char * completion, const int length);


// ============================
// clear state
// args: runtime handle
// returns: Error codes
int rwkvmobile_runtime_clear_state(rwkvmobile_runtime_t runtime);

#ifdef __cplusplus
}
#endif

#endif // C_API_H