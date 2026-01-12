#include <iostream>
#include <chrono>
#if _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif
#include "commondef.h"
#include "c_api.h"

#define ENSURE_SUCCESS_OR_LOG_EXIT(x, msg) if (x != rwkvmobile::RWKV_SUCCESS) { std::cout << msg << std::endl; return 1; }

char msg0[] = "Write a poem about the beauty of nature.";
char msg1[] = "What's the weather like today?";
char msg2[] = "Thanks!";

std::string response;
void callback(const char *msg, const int, const char *next) {
    // std::cout << "Callback: " << msg << std::endl;
    response = std::string(msg);
};

void test_get_loaded_models_info(rwkvmobile_runtime_t runtime) {
    std::cout << "\n=== Testing Get Loaded Models Info ===" << std::endl;

    // Test getting loaded model IDs list
    int model_ids[10];
    int count = rwkvmobile_runtime_get_loaded_model_ids(runtime, model_ids, 10);

    std::cout << "Number of loaded models: " << count << std::endl;
    std::cout << "Model ID list: ";
    for (int i = 0; i < count; i++) {
        std::cout << model_ids[i];
        if (i < count - 1) std::cout << ", ";
    }
    std::cout << std::endl;

    // Test getting detailed model info
    struct loaded_models_list models_list = rwkvmobile_runtime_get_loaded_models_info(runtime);

    std::cout << "\nDetailed model information:" << std::endl;
    for (int i = 0; i < models_list.count; i++) {
        struct model_info* model = &models_list.models[i];

        std::cout << "Model " << (i + 1) << ":" << std::endl;
        std::cout << "  Model ID: " << model->model_id << std::endl;
        std::cout << "  Model Path: " << (model->model_path ? model->model_path : "N/A") << std::endl;
        std::cout << "  Backend Name: " << (model->backend_name ? model->backend_name : "N/A") << std::endl;
        std::cout << "  Tokenizer Path: " << (model->tokenizer_path ? model->tokenizer_path : "N/A") << std::endl;
        std::cout << "  User Role: " << (model->user_role ? model->user_role : "N/A") << std::endl;
        std::cout << "  Response Role: " << (model->response_role ? model->response_role : "N/A") << std::endl;
        std::cout << "  BOS Token: " << (model->bos_token ? model->bos_token : "N/A") << std::endl;
        std::cout << "  EOS Token: " << (model->eos_token ? model->eos_token : "N/A") << std::endl;
        std::cout << "  Thinking Token: " << (model->thinking_token ? model->thinking_token : "N/A") << std::endl;
        std::cout << "  Is Generating: " << (model->is_generating ? "Yes" : "No") << std::endl;
        std::cout << "  Vocab Size: " << model->vocab_size << std::endl;
        std::cout << std::endl;
    }

    // Free memory
    rwkvmobile_runtime_free_loaded_models_list(models_list);
}

void custom_sleep(int seconds) {
#if _WIN32
    Sleep(seconds * 1000);
#else
    sleep(seconds);
#endif
}

int main(int argc, char **argv) {
    // set stdout to be unbuffered
    setvbuf(stdout, NULL, _IONBF, 0);

    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <vocab_file> <model_file> <backend> <model_file2> <backend2>" << std::endl;
        return 1;
    }

    // Create runtime
    rwkvmobile_runtime_t runtime = rwkvmobile_runtime_init();
    if (runtime == nullptr) {
        std::cerr << "Failed to initialize runtime" << std::endl;
        return 1;
    }

    std::cout << "Runtime initialized successfully" << std::endl;

    // Test before loading model (should return empty list)
    std::cout << "\n--- Testing Before Model Load ---" << std::endl;
    test_get_loaded_models_info(runtime);

    // Load first model
    std::cout << "Loading model 0..." << std::endl;
    int model_id0 = rwkvmobile_runtime_load_model(runtime, argv[2], argv[3], argv[1]);
    ENSURE_SUCCESS_OR_LOG_EXIT(model_id0 < 0 ? model_id0 : rwkvmobile::RWKV_SUCCESS, "Failed to load model");

    std::cout << "Model loaded successfully" << std::endl;

    // Test getting loaded models list
    test_get_loaded_models_info(runtime);

    std::cout << "Loading second model..." << std::endl;
    int model_id1 = rwkvmobile_runtime_load_model(runtime, argv[4], argv[5], argv[1]);
    ENSURE_SUCCESS_OR_LOG_EXIT(model_id1 < 0 ? model_id1 : rwkvmobile::RWKV_SUCCESS, "Failed to load model");
    std::cout << "Second model loaded successfully" << std::endl;
    test_get_loaded_models_info(runtime);

    std::cout << "Chatting with model 0..." << std::endl;
    char *input_list0[] = {msg0, nullptr, nullptr};
    rwkvmobile_runtime_eval_chat_with_history_async(runtime, model_id0, (const char **)input_list0, 1, 200, callback, false, false, true, FORCE_LANG_NONE);
    while (rwkvmobile_runtime_is_generating(runtime, model_id0)) {
        std::cout << ".";
        custom_sleep(1);
    }
    std::cout << std::endl;
    std::cout << "Response: " << response << std::endl;
    std::string response_str_0 = response;
    input_list0[1] = (char*)response_str_0.c_str();
    input_list0[2] = (char*)msg2;

    std::cout << "Chatting with model 1..." << std::endl;
    char *input_list1[] = {msg1, nullptr, nullptr};
    rwkvmobile_runtime_eval_chat_with_history_async(runtime, model_id1, (const char **)input_list1, 1, 200, callback, false, false, true, FORCE_LANG_NONE);
    while (rwkvmobile_runtime_is_generating(runtime, model_id1)) {
        std::cout << ".";
        custom_sleep(1);
    }
    std::cout << std::endl;
    std::cout << "Response: " << response << std::endl;
    std::string response_str_1 = response;
    input_list1[1] = (char*)response_str_1.c_str();
    input_list1[2] = (char*)msg2;

    std::cout << "Chatting again with model 0..." << std::endl;
    rwkvmobile_runtime_eval_chat_with_history_async(runtime, model_id0, (const char **)input_list0, 3, 200, callback, false, false, true, FORCE_LANG_NONE);
    while (rwkvmobile_runtime_is_generating(runtime, model_id0)) {
        std::cout << ".";
        custom_sleep(1);
    }
    std::cout << std::endl;
    std::cout << "Response: " << response << std::endl;

    std::cout << "Chatting again with model 1..." << std::endl;
    rwkvmobile_runtime_eval_chat_with_history_async(runtime, model_id1, (const char **)input_list1, 3, 200, callback, false, false, true, FORCE_LANG_NONE);
    while (rwkvmobile_runtime_is_generating(runtime, model_id1)) {
        std::cout << ".";
        custom_sleep(1);
    }
    std::cout << std::endl;
    std::cout << "Response: " << response << std::endl;

    // Release runtime
    rwkvmobile_runtime_release(runtime);
    std::cout << "\nTest completed!" << std::endl;

    return 0;
}
