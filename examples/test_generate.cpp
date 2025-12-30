#include <iostream>
#include <chrono>

#ifndef _WIN32
#include <unistd.h>
#endif

#include "commondef.h"
#include "runtime.h"

#define ENSURE_SUCCESS_OR_LOG_EXIT(x, msg) if (x != rwkvmobile::RWKV_SUCCESS) { std::cout << msg << std::endl; return 1; }

void callback(const char *msg, const int, const char *next) {
    std::cout << next;
}

int main(int argc, char **argv) {
    // set stdout to be unbuffered
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc != 4 && argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <vocab_file> <model_file> <backend> [prompt]" << std::endl;
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


    rwkvmobile::Runtime runtime;
    int model_id = runtime.load_model(argv[2], argv[3], argv[1], extra_data);
    ENSURE_SUCCESS_OR_LOG_EXIT(model_id < 0 ? model_id : rwkvmobile::RWKV_SUCCESS, "Failed to load model");
    if (model_id < 0) return 1;
    runtime.set_sampler_params(model_id, 1.0, 1, 1.0);
    runtime.set_penalty_params(model_id, 0.0, 0.0, 0.0);

    std::cout << "Generating demo text..." << std::endl;

    // std::string prompt = "User: Write me a poem about a cat\n\nAssistant:";
    std::string prompt = "The Eiffel Tower is in the city of";
    if (argc == 5) {
        prompt = argv[4];
    }
    std::cout << prompt;
    ENSURE_SUCCESS_OR_LOG_EXIT(runtime.gen_completion(model_id, prompt, 1000, 261, callback), "\nFailed to generate chat message");
    // std::cout << runtime.get_response_buffer_content(model_id);

    std::cout << std::endl;

    std::cout << "Prefill speed: " << runtime.get_avg_prefill_speed(model_id) << " tokens/s" << std::endl;
    std::cout << "Decode speed: " << runtime.get_avg_decode_speed(model_id) << " tokens/s" << std::endl;

    runtime.release();

    return 0;
}
