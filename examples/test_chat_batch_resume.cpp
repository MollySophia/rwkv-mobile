#include <iostream>
#include <string>
#include <vector>

#if _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "commondef.h"
#include "c_api.h"

#define ENSURE_SUCCESS_OR_LOG_EXIT(x, msg) if (x != rwkvmobile::RWKV_SUCCESS) { std::cout << msg << std::endl; return 1; }

void custom_sleep(int seconds) {
#if _WIN32
    Sleep(seconds * 1000);
#else
    sleep(seconds);
#endif
}

struct BatchInputs {
    std::vector<std::vector<const char *>> pointers;
    std::vector<const char **> outer;
    std::vector<int> lengths;
};

BatchInputs make_batch_inputs(const std::vector<std::vector<std::string>> &inputs) {
    BatchInputs batch;
    batch.pointers.resize(inputs.size());
    batch.outer.resize(inputs.size());
    batch.lengths.resize(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++) {
        batch.lengths[i] = (int)inputs[i].size();
        batch.pointers[i].reserve(inputs[i].size());
        for (const auto &message : inputs[i]) {
            batch.pointers[i].push_back(message.c_str());
        }
        batch.outer[i] = batch.pointers[i].data();
    }
    return batch;
}

void wait_for_generation(rwkvmobile_runtime_t runtime, int model_id, int stop_after_seconds) {
    int elapsed = 0;
    bool stop_requested = false;
    while (rwkvmobile_runtime_is_generating(runtime, model_id)) {
        std::cout << "Waiting for generation to finish..." << std::endl;
        custom_sleep(1);
        elapsed++;
        if (stop_after_seconds > 0 && elapsed > stop_after_seconds && !stop_requested) {
            std::cout << "Stopping batch generation..." << std::endl;
            rwkvmobile_runtime_stop_generation(runtime, model_id);
            stop_requested = true;
        }
    }
}

std::vector<std::string> get_batch_responses(rwkvmobile_runtime_t runtime, int model_id) {
    auto response_batch = rwkvmobile_runtime_get_response_buffer_content_batch(runtime, model_id);
    std::vector<std::string> responses;
    responses.reserve((size_t)response_batch.batch_size);
    for (int i = 0; i < response_batch.batch_size; i++) {
        responses.emplace_back(response_batch.contents[i] ? response_batch.contents[i] : "");
    }
    rwkvmobile_runtime_free_response_buffer_batch(response_batch);
    return responses;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <vocab_file> <model_file> <backend> <batch_size>" << std::endl;
        return 1;
    }

    int batch_size = atoi(argv[4]);
    if (batch_size <= 0) {
        std::cerr << "batch_size must be positive" << std::endl;
        return 1;
    }

    rwkvmobile_runtime_t runtime = rwkvmobile_runtime_init();
    int model_id = rwkvmobile_runtime_load_model(runtime, argv[2], argv[3], argv[1]);
    ENSURE_SUCCESS_OR_LOG_EXIT(model_id < 0 ? model_id : rwkvmobile::RWKV_SUCCESS, "Failed to load model");
    if (model_id < 0) return 1;

    rwkvmobile_runtime_set_sampler_params(runtime, model_id, {1.0, 1, 1.0});

    std::vector<std::string> questions = {
        "What's the weather like today?",
        "Write a short poem about rain.",
        "Give me one practical study tip.",
        "Explain why the sky is blue in one sentence.",
    };

    std::vector<std::vector<std::string>> first_inputs((size_t)batch_size);
    for (int i = 0; i < batch_size; i++) {
        first_inputs[(size_t)i] = {
            "Hello!",
            "Hello! I'm your AI assistant. I'm here to help.",
            questions[(size_t)i % questions.size()],
        };
    }

    std::cout << "Starting batch chat, then interrupting it..." << std::endl;
    auto first_batch = make_batch_inputs(first_inputs);
    ENSURE_SUCCESS_OR_LOG_EXIT(
        rwkvmobile_runtime_eval_chat_batch_with_history_async(
            runtime,
            model_id,
            (const char ***)first_batch.outer.data(),
            first_batch.lengths.data(),
            batch_size,
            80,
            nullptr,
            false,
            false,
            nullptr,
            true),
        "Failed to start batch chat");

    wait_for_generation(runtime, model_id, 2);
    auto partial_responses = get_batch_responses(runtime, model_id);
    for (int i = 0; i < batch_size; i++) {
        std::cout << "Partial response (batch " << i << "): " << partial_responses[(size_t)i] << std::endl;
    }

    std::vector<std::vector<std::string>> resume_inputs((size_t)batch_size);
    for (int i = 0; i < batch_size; i++) {
        resume_inputs[(size_t)i] = first_inputs[(size_t)i];
        resume_inputs[(size_t)i].push_back(partial_responses[(size_t)i]);
    }

    std::cout << "Resuming batch chat from partial assistant responses..." << std::endl;
    auto second_batch = make_batch_inputs(resume_inputs);
    ENSURE_SUCCESS_OR_LOG_EXIT(
        rwkvmobile_runtime_eval_chat_batch_with_history_async(
            runtime,
            model_id,
            (const char ***)second_batch.outer.data(),
            second_batch.lengths.data(),
            batch_size,
            80,
            nullptr,
            false,
            false,
            nullptr,
            false),
        "Failed to resume batch chat");

    wait_for_generation(runtime, model_id, 0);
    auto resumed_responses = get_batch_responses(runtime, model_id);
    for (int i = 0; i < batch_size; i++) {
        const auto &partial = partial_responses[(size_t)i];
        const auto &resumed = resumed_responses[(size_t)i];
        std::cout << "Resumed response (batch " << i << "): " << resumed << std::endl;
        if (!partial.empty() && resumed.rfind(partial, 0) != 0) {
            std::cerr << "Resumed response does not start with partial response for batch " << i << std::endl;
            rwkvmobile_runtime_release_model(runtime, model_id);
            rwkvmobile_runtime_release(runtime);
            return 1;
        }
    }

    rwkvmobile_runtime_release_model(runtime, model_id);
    rwkvmobile_runtime_release(runtime);

    return 0;
}
