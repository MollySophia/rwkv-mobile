#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

#include "commondef.h"
#include "runtime.h"

#define ENSURE_SUCCESS_OR_LOG_EXIT(x, msg) if (x != rwkvmobile::RWKV_SUCCESS) { std::cout << msg << std::endl; return 1; }

namespace {

std::vector<int> make_random_tokens(int count, int vocab_size, std::mt19937 &rng) {
    std::uniform_int_distribution<int> dist(0, std::max(0, vocab_size - 1));
    std::vector<int> ids(count);
    for (int &id : ids) {
        id = dist(rng);
    }
    return ids;
}

void cooldown_if_needed(int seconds) {
    if (seconds <= 0) {
        return;
    }
    std::cout << "Cooldown: sleeping for " << seconds << " seconds..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(seconds));
}

int benchmark_prefill(
    rwkvmobile::Runtime &runtime,
    int model_id,
    const std::vector<int> &prompt_ids,
    rwkvmobile::Tensor1D &logits
) {
    runtime.clear_state(model_id);
    runtime.reset_inference_speed_stats(model_id);

    ENSURE_SUCCESS_OR_LOG_EXIT(runtime.eval_logits(model_id, prompt_ids, logits), "Prefill warmup failed");

    runtime.clear_state(model_id);
    runtime.reset_inference_speed_stats(model_id);

    ENSURE_SUCCESS_OR_LOG_EXIT(runtime.eval_logits(model_id, prompt_ids, logits), "Prefill benchmark failed");
    std::cout << "Prefill speed (prompt_len=" << prompt_ids.size()
              << "): " << runtime.get_avg_prefill_speed(model_id) << " tokens/s" << std::endl;
    return 0;
}

int benchmark_decode_for_batch_size(
    rwkvmobile::Runtime &runtime,
    int model_id,
    int vocab_size,
    int batch_size,
    int decode_steps,
    std::mt19937 &rng,
    rwkvmobile::Tensor1D &logits
) {
    runtime.clear_state(model_id);
    runtime.reset_inference_speed_stats(model_id);

    std::vector<int> ids = make_random_tokens(batch_size, vocab_size, rng);
    for (int i = 0; i < 8; ++i) {
        ids = make_random_tokens(batch_size, vocab_size, rng);
        if (batch_size == 1) {
            ENSURE_SUCCESS_OR_LOG_EXIT(runtime.eval_logits(model_id, ids[0], logits), "Decode warmup failed");
        } else {
            ENSURE_SUCCESS_OR_LOG_EXIT(runtime.eval_logits_batch_decode(model_id, ids, logits), "Batch decode warmup failed");
        }
    }

    runtime.clear_state(model_id);
    runtime.reset_inference_speed_stats(model_id);

    for (int i = 0; i < decode_steps; ++i) {
        ids = make_random_tokens(batch_size, vocab_size, rng);
        if (batch_size == 1) {
            ENSURE_SUCCESS_OR_LOG_EXIT(runtime.eval_logits(model_id, ids[0], logits), "Decode benchmark failed");
        } else {
            ENSURE_SUCCESS_OR_LOG_EXIT(runtime.eval_logits_batch_decode(model_id, ids, logits), "Batch decode benchmark failed");
        }
    }

    std::cout << "Decode speed (bsz=" << batch_size
              << ", steps=" << decode_steps << "): "
              << runtime.get_avg_decode_speed(model_id) << " tokens/s" << std::endl;
    return 0;
}

} // namespace

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc < 3 || argc > 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <model_file> <backend> [prompt_len=512] [decode_steps=128] [cooldown_seconds=5]"
                  << std::endl;
        return 1;
    }

    const int prompt_len = (argc >= 4) ? std::max(1, atoi(argv[3])) : 512;
    const int decode_steps = (argc >= 5) ? std::max(1, atoi(argv[4])) : 128;
    const int cooldown_seconds = (argc >= 6) ? std::max(0, atoi(argv[5])) : 5;

    rwkvmobile::Runtime runtime;
    int model_id = runtime.load_model(argv[1], argv[2], "", nullptr);
    ENSURE_SUCCESS_OR_LOG_EXIT(model_id < 0 ? model_id : rwkvmobile::RWKV_SUCCESS, "Failed to load model");
    if (model_id < 0) return 1;

    const int vocab_size = runtime.get_vocab_size(model_id);
    if (vocab_size <= 0) {
        std::cerr << "Invalid vocab size: " << vocab_size << std::endl;
        runtime.release();
        return 1;
    }

    std::random_device rd;
    std::mt19937 rng(rd());
    rwkvmobile::Tensor1D logits;

    std::vector<int> prompt_ids = make_random_tokens(prompt_len, vocab_size, rng);

    std::vector<int> supported_batch_sizes = runtime.get_supported_batch_sizes(model_id);
    supported_batch_sizes.erase(
        std::remove_if(
            supported_batch_sizes.begin(),
            supported_batch_sizes.end(),
            [](int bsz) { return bsz <= 1; }
        ),
        supported_batch_sizes.end()
    );
    std::sort(supported_batch_sizes.begin(), supported_batch_sizes.end());
    supported_batch_sizes.erase(
        std::unique(supported_batch_sizes.begin(), supported_batch_sizes.end()),
        supported_batch_sizes.end()
    );
    supported_batch_sizes.insert(supported_batch_sizes.begin(), 1);

    std::cout << "Model: " << argv[1] << std::endl;
    std::cout << "Backend: " << argv[2] << std::endl;
    std::cout << "Prompt length: " << prompt_len << std::endl;
    std::cout << "Decode steps: " << decode_steps << std::endl;
    std::cout << "Cooldown seconds: " << cooldown_seconds << std::endl;
    std::cout << "Decode batch sizes:";
    for (int bsz : supported_batch_sizes) {
        std::cout << " " << bsz;
    }
    std::cout << std::endl;

    int ret = benchmark_prefill(runtime, model_id, prompt_ids, logits);
    if (ret != 0) {
        runtime.release();
        return ret;
    }

    for (size_t i = 0; i < supported_batch_sizes.size(); ++i) {
        cooldown_if_needed(cooldown_seconds);
        ret = benchmark_decode_for_batch_size(
            runtime,
            model_id,
            vocab_size,
            supported_batch_sizes[i],
            decode_steps,
            rng,
            logits
        );
        if (ret != 0) {
            runtime.release();
            return ret;
        }
    }

    runtime.release();
    return 0;
}
