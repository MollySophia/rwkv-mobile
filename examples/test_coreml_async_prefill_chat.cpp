#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "c_api.h"
#include "commondef.h"
#include "runtime.h"

#define ENSURE_SUCCESS_OR_LOG_EXIT(x, msg) \
    if ((x) != rwkvmobile::RWKV_SUCCESS) { \
        std::cout << msg << std::endl; \
        return 1; \
    }

static void quiet_callback(const char *, const int, const char *) {
}

static std::string make_medium_prompt(int round) {
    return "Round " + std::to_string(round) + ". "
        "You are evaluating a mobile RWKV runtime. Summarize the following deployment notes in two concise paragraphs: "
        "the Core ML backend can load the decode function first, return control to the application, and keep loading the "
        "prefill function in the background. Before prefill is ready, long prompts are processed through repeated decode "
        "steps, so the measured prefill speed should be noticeably lower. After the background prefill function finishes "
        "loading, the same medium-length prompt path should switch to the prefill graph and the reported prefill speed "
        "should jump. Keep the answer short and mention only runtime behavior, loading state, and user-visible latency.";
}

int main(int argc, char **argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 4 || argc > 7) {
        std::cerr << "Usage: " << argv[0] << " <vocab_file> <model_dir> <rounds> [max_tokens] [sleep_ms] [threshold_ms]" << std::endl;
        std::cerr << "threshold_ms: 0 uses default 5000, positive overrides it, negative forces async" << std::endl;
        std::cerr << "Example: " << argv[0] << " assets/rwkv_vocab_v20230424.txt /path/to/coreml_model_dir 12 8 1000 5000" << std::endl;
        return 1;
    }

    const char *vocab_path = argv[1];
    const char *model_path = argv[2];
    const int rounds = std::max(1, std::atoi(argv[3]));
    const int max_tokens = argc >= 5 ? std::max(1, std::atoi(argv[4])) : 8;
    const int sleep_ms = argc >= 6 ? std::max(0, std::atoi(argv[5])) : 1000;
    const int threshold_ms = argc >= 7 ? std::atoi(argv[6]) : 0;

    rwkvmobile::Runtime runtime;
    coreml_args args{};
    args.load_prefill_async = 1;
    args.async_prefill_decode_load_threshold_ms = threshold_ms;

    auto load_start = std::chrono::steady_clock::now();
    int model_id = runtime.load_model(model_path, "coreml", vocab_path, &args);
    auto load_end = std::chrono::steady_clock::now();
    ENSURE_SUCCESS_OR_LOG_EXIT(model_id < 0 ? model_id : rwkvmobile::RWKV_SUCCESS, "Failed to load CoreML model");
    if (model_id < 0) return 1;

    const double load_ms = std::chrono::duration<double, std::milli>(load_end - load_start).count();
    std::cout << "Decode-first load returned in " << load_ms << " ms" << std::endl;
    std::cout << "async prefill threshold_ms=" << threshold_ms << " (0 means default)" << std::endl;
    std::cout << "CoreML will either print \"CoreML async prefill ready\" or skip async by threshold." << std::endl;

    runtime.set_sampler_params(model_id, 1.0f, 1, 1.0f);
    runtime.set_penalty_params(model_id, 0.0f, 0.0f, 0.0f);

    for (int round = 1; round <= rounds; ++round) {
        std::vector<std::string> input = {make_medium_prompt(round)};
        auto start = std::chrono::steady_clock::now();
        int ret = runtime.chat(
            model_id,
            input,
            max_tokens,
            quiet_callback,
            false,
            false,
            true
        );
        auto end = std::chrono::steady_clock::now();
        ENSURE_SUCCESS_OR_LOG_EXIT(ret, "chat failed");

        const double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "round " << round
                  << ": elapsed=" << elapsed_ms << " ms"
                  << ", prefill=" << runtime.get_avg_prefill_speed(model_id) << " tok/s"
                  << ", decode=" << runtime.get_avg_decode_speed(model_id) << " tok/s"
                  << std::endl;

        runtime.clear_state(model_id);
        if (sleep_ms > 0 && round < rounds) {
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
        }
    }

    runtime.release();
    return 0;
}
