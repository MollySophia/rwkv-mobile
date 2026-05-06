#include <iostream>
#include <string>
#include <vector>

#include "commondef.h"
#include "runtime.h"

#define ENSURE_SUCCESS_OR_LOG_EXIT(x, msg) \
    if ((x) != rwkvmobile::RWKV_SUCCESS) { \
        std::cerr << msg << std::endl; \
        return 1; \
    }

int main(int argc, char **argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 5 || argc > 7) {
        std::cerr << "Usage: " << argv[0]
                  << " <vocab_file> <state_pth_file> <model_file> <backend> [prompt] [max_tokens]"
                  << std::endl;
        return 1;
    }

    const std::string vocab_path = argv[1];
    const std::string state_path = argv[2];
    const std::string model_path = argv[3];
    const std::string backend = argv[4];
    const std::string prompt = argc > 5 ? argv[5] : "User: hello\n\nAssistant:";
    const int max_tokens = argc > 6 ? std::stoi(argv[6]) : 64;

    rwkvmobile::Runtime runtime;
    const int model_id = runtime.load_model(model_path, backend, vocab_path, nullptr);
    ENSURE_SUCCESS_OR_LOG_EXIT(model_id < 0 ? model_id : rwkvmobile::RWKV_SUCCESS, "Failed to load model");

    runtime.set_sampler_params(model_id, 1.0f, 1, 1.0f);
    runtime.set_penalty_params(model_id, 0.0f, 0.0f, 0.996f);

    const int ret = runtime.load_initial_state(model_id, state_path);
    ENSURE_SUCCESS_OR_LOG_EXIT(ret, "Failed to load pth initial state");

    const std::string state_prompt = "<state src=\"" + state_path + "\">" + prompt;
    runtime.set_prompt(model_id, state_prompt);
    runtime.chat(model_id, {}, max_tokens, nullptr);

    std::cout << "Response: " << runtime.get_response_buffer_content(model_id) << std::endl;
    runtime.release();
    return 0;
}
