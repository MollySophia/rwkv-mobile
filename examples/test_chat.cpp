#include <iostream>
#include <chrono>

#ifndef _WIN32
#include <unistd.h>
#endif

#include "commondef.h"
#include "runtime.h"
#include "logger.h"

#define ENSURE_SUCCESS_OR_LOG_EXIT(x, msg) if (x != rwkvmobile::RWKV_SUCCESS) { std::cout << msg << std::endl; return 1; }

void callback(const char *msg, const int, const char *next) {
    std::cout << next;
}

int main(int argc, char **argv) {
    // set stdout to be unbuffered
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc != 5 && argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <vocab_file> <model_file> <backend> <prompt> [use_reasoning]" << std::endl;
        std::cerr << "use_reasoning: 0 or 1 (defaults to 1)" << std::endl;
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

    bool use_reasoning = true;
    if (argc == 6) {
        use_reasoning = atoi(argv[5]) == 1;
    }

    rwkvmobile::Runtime runtime;
    int model_id = runtime.load_model(argv[2], argv[3], argv[1], extra_data);
    ENSURE_SUCCESS_OR_LOG_EXIT(model_id < 0 ? model_id : rwkvmobile::RWKV_SUCCESS, "Failed to load model");
    if (model_id < 0) return 1;
    // runtime.set_sampler_params(model_id, 1.0, 1, 1.0);

    std::cout << "User: " << argv[4] << "\n\nAssistant:";
    if (use_reasoning) {
        std::cout << " " << runtime.get_thinking_token(model_id);
    }
    std::vector<std::string> input_list = {
        argv[4],
    };
    runtime.chat(model_id, input_list, 1024, callback, use_reasoning, false, true);
    std::cout << std::endl;

    std::cout << "Prefill speed: " << runtime.get_avg_prefill_speed(model_id) << " tokens/s" << std::endl;
    std::cout << "Decode speed: " << runtime.get_avg_decode_speed(model_id) << " tokens/s" << std::endl;

    runtime.release();

    return 0;
}
