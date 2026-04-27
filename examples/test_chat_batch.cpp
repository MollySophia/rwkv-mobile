#include <iostream>
#include <chrono>
#include <string>
#include <utility>
#include <vector>

#include "commondef.h"
#include "runtime.h"
#include "logger.h"

#define ENSURE_SUCCESS_OR_LOG_EXIT(x, msg) if (x != rwkvmobile::RWKV_SUCCESS) { std::cout << msg << std::endl; return 1; }

int main(int argc, char **argv) {
    // set stdout to be unbuffered
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc != 5 && argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <vocab_file> <model_file> <backend> <batch_size> [style_repro]" << std::endl;
        return 1;
    }

    rwkvmobile::Runtime runtime;
    int model_id = runtime.load_model(argv[2], argv[3], argv[1], nullptr);
    ENSURE_SUCCESS_OR_LOG_EXIT(model_id < 0 ? model_id : rwkvmobile::RWKV_SUCCESS, "Failed to load model");
    if (model_id < 0) return 1;

    int batch_size = atoi(argv[4]);

    if (argc == 6 && std::string(argv[5]) == "style_repro") {
        if (batch_size != 4) {
            std::cerr << "style_repro expects batch_size=4" << std::endl;
            runtime.release();
            return 1;
        }
        runtime.set_seed(model_id, 42);
        runtime.set_sampler_params(model_id, 1.0f, 1, 1.0f);

        const std::string base_prompt = "用三句话介绍一下杭州西湖。";
        const std::vector<std::pair<std::string, std::string>> styles = {
            {"gu", " 请用文言文回答。"},
            {"mao", " 请用可爱的猫咪口吻回答，多使用“喵”，保持猫风格。"},
            {"en", " Use English only. Direct answer. No preface. Never speak in Chinese. Do not use any Chinese characters."},
            {"ja", " 日本語のみ。前置きなしで直接回答。"},
        };

        std::vector<std::vector<std::string>> input_list_batch;
        input_list_batch.reserve(styles.size());
        for (const auto &style : styles) {
            input_list_batch.push_back({base_prompt + style.second});
        }

        std::cout << "Batch style prompts:" << std::endl;
        ENSURE_SUCCESS_OR_LOG_EXIT(runtime.chat_batch(model_id, input_list_batch, 120, batch_size, nullptr, false, false, true), "Failed to chat batch");
        auto batch_response = runtime.get_response_buffer_content_batch(model_id);
        for (int i = 0; i < batch_size; i++) {
            std::cout << "[" << styles[i].first << "] " << batch_response[i] << std::endl << std::endl;
        }

        std::cout << "Single EN prompt:" << std::endl;
        ENSURE_SUCCESS_OR_LOG_EXIT(runtime.chat(model_id, {base_prompt + styles[2].second}, 120, nullptr, false, false, true), "Failed to chat single EN");
        std::cout << runtime.get_response_buffer_content(model_id) << std::endl << std::endl;

        runtime.release();
        return 0;
    }

    std::vector<std::string> input_list = {
        "Hello!",
        "Hello! I'm your AI assistant. I'm here to help you with various tasks, such as answering questions, brainstorming ideas, drafting emails, writing code, providing advice, and much more.",
        "随机说一个两位数",
    };
    std::vector<std::vector<std::string>> input_list_batch(batch_size);
    for (int i = 0; i < batch_size; i++) {
        input_list_batch[i] = input_list;
    }
    std::cout << "Testing batch chat prompt: " << input_list[input_list.size()-1] << std::endl << std::endl;
    ENSURE_SUCCESS_OR_LOG_EXIT(runtime.chat_batch(model_id, input_list_batch, 300, batch_size, nullptr, false, false, true), "Failed to chat batch");
    auto batch_response = runtime.get_response_buffer_content_batch(model_id);
    for (int i = 0; i < batch_size; i++) {
        std::cout << "Response " << i << ": " << batch_response[i] << std::endl << std::endl;
    }

    input_list.push_back(batch_response[rand() % batch_size]);
    input_list.push_back("复述一遍你刚才说的两位数");

    std::cout << "Testing new chat prompt: " << input_list[input_list.size()-1] << std::endl << std::endl;
    ENSURE_SUCCESS_OR_LOG_EXIT(runtime.chat(model_id, input_list, 300, nullptr, false), "Failed to chat");
    std::cout << "Response: " << runtime.get_response_buffer_content(model_id) << std::endl;

    runtime.release();

    return 0;
}
