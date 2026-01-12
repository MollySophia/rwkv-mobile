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

std::string response;
void callback(const char *msg, const int, const char *next) {
    // std::cout << "Callback: " << msg << std::endl;
    response = std::string(msg);
};

char msg0[] = "Hello!";
char msg1[] = "Hello! I'm your AI assistant. I'm here to help you with various tasks, such as answering questions, brainstorming ideas, drafting emails, writing code, providing advice, and much more.";
char msg2[] = "What's the weather like today?";
char msg3[] = "Write me a poem";

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
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <vocab_file> <model_file> <backend>" << std::endl;
        return 1;
    }

    rwkvmobile_runtime_t runtime = rwkvmobile_runtime_init();
    int model_id = rwkvmobile_runtime_load_model(runtime, argv[2], argv[3], argv[1]);
    ENSURE_SUCCESS_OR_LOG_EXIT(model_id < 0 ? model_id : rwkvmobile::RWKV_SUCCESS, "Failed to load model");
    rwkvmobile_runtime_set_sampler_params(runtime, model_id, {1.0, 1, 1.0});

    std::cout << "Testing original input list" << std::endl;
    char *input_list[] = {
        msg0,
        msg1,
        msg2,
        nullptr
    };

    rwkvmobile_runtime_eval_chat_with_history_async(runtime, model_id, (const char **)input_list, 3, 50, callback, false, false, true, FORCE_LANG_NONE);
    int count = 0;
    while (rwkvmobile_runtime_is_generating(runtime, model_id)) {
        std::cout << "Waiting for generation to finish..." << std::endl;
        custom_sleep(1);
        count++;
        if (count > 2) {
            rwkvmobile_runtime_stop_generation(runtime, model_id);
            break;
        }
    }
    std::cout << "Response: " << response << std::endl;

    for (int i = 0; i < 10; i++) {
        custom_sleep(1);
    }
    input_list[3] = (char*)response.c_str();
    rwkvmobile_runtime_eval_chat_with_history_async(runtime, model_id, (const char **)input_list, 4, 50, callback, false, false, true, FORCE_LANG_NONE);
    while (rwkvmobile_runtime_is_generating(runtime, model_id)) {
        std::cout << "Waiting for generation to finish..." << std::endl;
        custom_sleep(1);
    }
    std::cout << "Response: " << response << std::endl;

    rwkvmobile_runtime_release_model(runtime, model_id);
    rwkvmobile_runtime_release(runtime);

    return 0;
}
