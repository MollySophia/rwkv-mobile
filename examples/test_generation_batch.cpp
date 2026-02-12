#include <iostream>
#include <chrono>

#ifndef _WIN32
#include <unistd.h>
#else
#include <windows.h>
#endif

#include "commondef.h"
#include "c_api.h"
#include <vector>
#include <cstring>

#define ENSURE_SUCCESS_OR_LOG_EXIT(x, msg) if (x != rwkvmobile::RWKV_SUCCESS) { std::cout << msg << std::endl; return 1; }

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
    if (argc != 5 && argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <vocab_file> <model_file> <backend> <batch_size> [prompt]" << std::endl;
        return 1;
    }

    void *extra_data = nullptr;

#ifndef _WIN32
    std::string path;
    if (strcmp(argv[3], "qnn") == 0) {
        char *buffer;
        if ((buffer = getcwd(NULL, 0)) == NULL) {
            perror("getcwd error");
        }
        path = std::string(buffer);
        setenv("LD_LIBRARY_PATH", path.c_str(), 1);
        setenv("ADSP_LIBRARY_PATH", path.c_str(), 1);
        if (buffer) {
            free(buffer);
        }
        std::cout << "cwd: " << path << std::endl;
        path = path + "/libQnnHtp.so";
        extra_data = (void *)path.c_str();
    }
#endif

    rwkvmobile_runtime_t runtime = rwkvmobile_runtime_init();
    int model_id = rwkvmobile_runtime_load_model_with_extra(runtime, argv[2], argv[3], argv[1], extra_data);
    ENSURE_SUCCESS_OR_LOG_EXIT(model_id < 0 ? model_id : rwkvmobile::RWKV_SUCCESS, "Failed to load model");
    if (model_id < 0) return 1;
    rwkvmobile_runtime_set_sampler_params(runtime, model_id, {1.0, 1, 1.0});
    rwkvmobile_runtime_set_penalty_params(runtime, model_id, {0.0, 0.0, 0.0});

    std::cout << "Generating demo text..." << std::endl;

    // std::string prompt = "User: Write me a poem about a cat\n\nAssistant:";
    std::string prompt = "The Eiffel Tower is in the city of";
    int batch_size = atoi(argv[4]);
    if (argc == 6) {
        prompt = argv[5];
    }
    std::vector<const char *> prompts;
    for (int i = 0; i < batch_size; i++) {
        prompts.push_back(prompt.c_str());
    }
    std::cout << prompt;
    ENSURE_SUCCESS_OR_LOG_EXIT(rwkvmobile_runtime_gen_completion_batch_async(runtime, model_id, (const char **)prompts.data(), batch_size, 50, 261, nullptr, false), "\nFailed to generate chat message");

    std::cout << "Waiting for generation to finish...";
    while (rwkvmobile_runtime_is_generating(runtime, model_id)) {
        custom_sleep(1);
        std::cout << ".";
    }
    std::cout << " done" << std::endl;
    auto responses = rwkvmobile_runtime_get_response_buffer_content_batch(runtime, model_id);
    for (int i = 0; i < batch_size; i++) {
        std::cout << "Response (batch " << i << "): " << responses.contents[i] << std::endl << std::endl;
    }
    rwkvmobile_runtime_free_response_buffer_batch(responses);

    std::cout << std::endl;

    std::cout << "Prefill speed: " << rwkvmobile_runtime_get_avg_prefill_speed(runtime, model_id) << " tokens/s" << std::endl;
    std::cout << "Decode speed: " << rwkvmobile_runtime_get_avg_decode_speed(runtime, model_id) << " tokens/s" << std::endl;

    rwkvmobile_runtime_release(runtime);

    return 0;
}
