#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <iostream>
#include <algorithm>

#include "commondef.h"
#include "runtime.h"

#define ENSURE_SUCCESS_OR_LOG_EXIT(x, msg) if (x != rwkvmobile::RWKV_SUCCESS) { std::cout << msg << std::endl; return 1; }

int main(int argc, char **argv) {
    std::cout.setf(std::ios::unitbuf);
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <tokenizer_path> <model_path> <backend> <text_path>\n";
        return 1;
    }

    std::string tokenizer_path = argv[1];
    std::string model_path = argv[2];
    std::string backend = argv[3];
    std::string text_path = argv[4];

    rwkvmobile::Runtime runtime;
    int model_id = runtime.load_model(model_path, backend, tokenizer_path, nullptr); 
    ENSURE_SUCCESS_OR_LOG_EXIT(model_id < 0 ? model_id : rwkvmobile::RWKV_SUCCESS, "Failed to load model");
    if (model_id < 0) return 1;

    char *eval_text_buf;
    std::ifstream eval_text_file(text_path, std::ios::binary | std::ios::ate);
    size_t file_size;
    if (eval_text_file.is_open()) {
        eval_text_file.seekg(0, std::ios::end);
        file_size = eval_text_file.tellg();
        eval_text_buf = new char[file_size];
        eval_text_file.seekg(0, std::ios::beg);
        eval_text_file.read(eval_text_buf, file_size);
        eval_text_file.close();
    } else {
        std::cerr << "Unable to open file\n";
        return 1;
    }
    std::vector<std::string> eval_text;
    size_t next = 0;
    for (size_t i = 0; i < file_size; i++) {
        if (eval_text_buf[i] == '|') {
            eval_text.push_back(std::string(eval_text_buf + next, i - next));
            next = i + 1;
        }
    }
    delete[] eval_text_buf;
    std::cout << "Eval texts num: " << eval_text.size() << std::endl;

    float xsum = 0;
    int xcnt = 0;
    int xacc = 0;

    for (const auto &text : eval_text) {
        std::cout << "Sample num: " << xcnt << std::endl;
        auto prompt = text.substr(0, text.find_last_of(' '));
        auto target = text.substr(text.find_last_of(' '));
        std::cout << "Prompt: " << prompt << std::endl;
        std::cout << "Target: " << target << std::endl;
        std::cout << "Response: ";

        bool correct = false;
        float logits_val = -1e9f;
        std::string output_text;
        runtime.run_evaluation(model_id, prompt, target, correct, logits_val, output_text, true);
        std::cout << output_text << std::endl;

        xcnt++;
        if (correct) {
          xacc++;
        } 
        xsum += logits_val;

        // if (xcnt % 10 == 0) {
          std::cout << "\nAccuracy: " << xacc << "/" << xcnt << " = " << (float)xacc / xcnt << std::endl;
          std::cout << "Perplexity: " << std::exp(-xsum / xcnt) << std::endl;
          std::cout << "====================================\n";
        // }
    }

    runtime.release_model(model_id);
    runtime.release();
    return 0;
}