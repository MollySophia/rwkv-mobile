#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include "commondef.h"
#include "runtime.h"

#define ENSURE_SUCCESS_OR_LOG_EXIT(x, msg) \
    if ((x) != rwkvmobile::RWKV_SUCCESS) { \
        std::cout << (msg) << std::endl; \
        return 1; \
    }

namespace {

std::vector<int> make_random_tokens(int count, int vocab_size, std::mt19937 &rng) {
    std::uniform_int_distribution<int> dist(0, std::max(0, vocab_size - 1));
    std::vector<int> ids(count);
    for (int &id : ids) {
        id = dist(rng);
    }
    return ids;
}

uint32_t benchmark_seed() {
    const char *value = std::getenv("RWKV_BENCHMARK_SEED");
    if (value == nullptr || value[0] == '\0') {
        std::random_device rd;
        return rd();
    }
    return static_cast<uint32_t>(std::strtoul(value, nullptr, 10));
}

} // namespace

int main(int argc, char **argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 3 || argc > 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <model_file> <backend> [decode_steps=16] [warmup_steps=4]"
                  << std::endl;
        return 1;
    }

    const int decode_steps = (argc >= 4) ? std::max(1, std::atoi(argv[3])) : 16;
    const int warmup_steps = (argc >= 5) ? std::max(0, std::atoi(argv[4])) : 4;

    rwkvmobile::Runtime runtime;
    int model_id = runtime.load_model(argv[1], argv[2], "", nullptr);
    ENSURE_SUCCESS_OR_LOG_EXIT(model_id < 0 ? model_id : rwkvmobile::RWKV_SUCCESS, "Failed to load model");
    if (model_id < 0) {
        return 1;
    }

    const int vocab_size = runtime.get_vocab_size(model_id);
    if (vocab_size <= 0) {
        std::cerr << "Invalid vocab size: " << vocab_size << std::endl;
        runtime.release();
        return 1;
    }

    const uint32_t seed = benchmark_seed();
    std::mt19937 rng(seed);
    rwkvmobile::Tensor1D logits;

    std::cout << "Model: " << argv[1] << std::endl;
    std::cout << "Backend: " << argv[2] << std::endl;
    std::cout << "Decode steps: " << decode_steps << std::endl;
    std::cout << "Warmup steps: " << warmup_steps << std::endl;
    std::cout << "Seed: " << seed << std::endl;

    runtime.clear_state(model_id);
    runtime.reset_inference_speed_stats(model_id);
    std::vector<int> ids = make_random_tokens(warmup_steps + decode_steps, vocab_size, rng);
    for (int i = 0; i < warmup_steps; ++i) {
        ENSURE_SUCCESS_OR_LOG_EXIT(runtime.eval_logits(model_id, ids[i], logits), "Decode warmup failed");
    }

    runtime.clear_state(model_id);
    runtime.reset_inference_speed_stats(model_id);
    for (int i = 0; i < decode_steps; ++i) {
        ENSURE_SUCCESS_OR_LOG_EXIT(
            runtime.eval_logits(model_id, ids[warmup_steps + i], logits),
            "Decode benchmark failed"
        );
    }

    std::cout << "Decode speed: " << runtime.get_avg_decode_speed(model_id) << " tokens/s" << std::endl;
    runtime.release();
    return 0;
}
