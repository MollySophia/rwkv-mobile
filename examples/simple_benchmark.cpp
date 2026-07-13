#include <iostream>
#include <chrono>
#include <cstdlib>
#include <random>
#include <vector>

#include "commondef.h"
#include "runtime.h"

#define ENSURE_SUCCESS_OR_LOG_EXIT(x, msg) if (x != rwkvmobile::RWKV_SUCCESS) { std::cout << msg << std::endl; return 1; }

int main(int argc, char **argv) {
    // set stdout to be unbuffered
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc != 3 && argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <model_file> <backend> [prompt_tokens]" << std::endl;
        return 1;
    }

    rwkvmobile::Runtime runtime;
    int model_id = runtime.load_model(argv[1], argv[2], "", nullptr);
    ENSURE_SUCCESS_OR_LOG_EXIT(model_id < 0 ? model_id : rwkvmobile::RWKV_SUCCESS, "Failed to load model");
    if (model_id < 0) return 1;

    int vocab_size = runtime.get_vocab_size(model_id);
    int prompt_tokens = 512;
    if (argc == 4) {
        prompt_tokens = std::atoi(argv[3]);
        if (prompt_tokens <= 0) {
            std::cerr << "prompt_tokens must be positive" << std::endl;
            return 1;
        }
    }

    std::vector<int> prompt_ids((size_t)prompt_tokens);
    rwkvmobile::Tensor1D logits;

    for (int i = 0; i < prompt_tokens; i++) {
        prompt_ids[i] = rand() % vocab_size;
    }

    runtime.clear_state(model_id);
    runtime.reset_inference_speed_stats(model_id);

    // Warm up kernels and graph compilation on the same runtime instance.
    runtime.eval_logits(model_id, prompt_ids, logits);
    for (int i = 0; i < 128; i++) {
        runtime.eval_logits(model_id, rand() % vocab_size, logits);
    }

    runtime.clear_state(model_id);
    runtime.reset_inference_speed_stats(model_id);

    runtime.eval_logits(model_id, prompt_ids, logits);

    std::cout << "Prompt tokens: " << prompt_tokens << std::endl;
    std::cout << "Prefill speed: " << runtime.get_avg_prefill_speed(model_id) << " tokens/s" << std::endl;

    for (int i = 0; i < 128; i++) {
        runtime.eval_logits(model_id, rand() % vocab_size, logits);
    }
    std::cout << "Decode speed: " << runtime.get_avg_decode_speed(model_id) << " tokens/s" << std::endl;

    runtime.release();

    return 0;
}
