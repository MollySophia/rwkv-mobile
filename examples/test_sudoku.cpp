#include <iostream>
#include <chrono>

#include "commondef.h"
#include "runtime.h"
#include "c_api.h"

#define ENSURE_SUCCESS_OR_LOG_EXIT(x, msg) if (x != rwkvmobile::RWKV_SUCCESS) { std::cout << msg << std::endl; return 1; }

int current_length = 0;
long long num_tokens = 0;
void callback(const char *msg, const int, const char *next) {
    std::string msg_str(msg);
    std::cout << msg_str.substr(current_length);
    current_length = msg_str.length();
    num_tokens += 1;
    if (num_tokens % 1000 == 0) {
        std::cout << "\n\n Tokens: " << num_tokens << std::endl << std::endl;
    }
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
    std::cout << "Loaded model" << std::endl;
    rwkvmobile_runtime_set_penalty_params(runtime, model_id, {0, 0, 0});
    rwkvmobile_runtime_set_sampler_params(runtime, model_id, {1.0, 1, 1.0});

    std::string prompt = "<input>\n"
                        "8 0 0 0 0 0 0 0 0 \n"
                        "0 0 3 6 0 0 0 0 0 \n"
                        "0 7 0 0 9 0 2 0 0 \n"
                        "0 5 0 0 0 7 0 0 0 \n"
                        "0 0 0 0 4 5 7 0 0 \n"
                        "0 0 0 1 0 0 0 3 0 \n"
                        "0 0 1 0 0 0 0 6 8 \n"
                        "0 0 8 5 0 0 0 1 0 \n"
                        "0 9 0 0 0 0 4 0 0 \n"
                        "</input>\n\n";

    std::cout << std::endl;

    std::cout << "Generating completion" << std::endl;
    rwkvmobile_runtime_gen_completion(runtime, model_id, prompt.c_str(), 5000000, 105, callback, true);
    return 0;
}
