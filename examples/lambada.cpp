#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <iostream>
#include <algorithm>

#include "commondef.h"
#include "runtime.h"
#include "tensor.h"

#define ENSURE_SUCCESS_OR_LOG_EXIT(x, msg) if (x != rwkvmobile::RWKV_SUCCESS) { std::cout << msg << std::endl; return 1; }

static int run_evaluation_decode_only(
    rwkvmobile::Runtime& runtime,
    int model_id,
    const std::string& source_text,
    const std::string& target_text,
    bool& correct,
    float& logits_val,
    std::string& output_text,
    bool insert_bos_token
) {
    auto source_ids = runtime.tokenizer_encode(model_id, source_text);
    auto target_ids = runtime.tokenizer_encode(model_id, target_text);
    if (insert_bos_token) {
        source_ids.insert(source_ids.begin(), 0);
    }

    rwkvmobile::Tensor1D logits;
    int ret = runtime.clear_state(model_id);
    if (ret != rwkvmobile::RWKV_SUCCESS) {
        return ret;
    }
    for (int id : source_ids) {
        ret = runtime.eval_logits(model_id, id, logits);
        if (ret != rwkvmobile::RWKV_SUCCESS || logits.data_ptr == nullptr) {
            return ret ? ret : rwkvmobile::RWKV_ERROR_RUNTIME;
        }
    }

    correct = true;
    logits_val = 0.0f;
    std::vector<int> output_ids;
    output_ids.reserve(target_ids.size());
    for (size_t i = 0; i < target_ids.size(); ++i) {
        if (logits.data_ptr == nullptr || logits.count == 0) {
            return rwkvmobile::RWKV_ERROR_RUNTIME | rwkvmobile::RWKV_ERROR_INVALID_PARAMETERS;
        }

        int output_id = 0;
        float max_val = rwkvmobile::tensor1d_get_f32(logits, 0);
        for (size_t j = 1; j < logits.count; ++j) {
            const float v = rwkvmobile::tensor1d_get_f32(logits, j);
            if (v > max_val) {
                max_val = v;
                output_id = (int)j;
            }
        }

        double sum = 0.0;
        for (size_t j = 0; j < logits.count; ++j) {
            sum += std::exp((double)rwkvmobile::tensor1d_get_f32(logits, j) - (double)max_val);
        }
        const int target_id = target_ids[i];
        const double target_prob = std::exp((double)rwkvmobile::tensor1d_get_f32(logits, (size_t)target_id) - (double)max_val) / sum;
        logits_val += (float)std::log(std::max(target_prob, 1e-45));

        output_ids.push_back(output_id);
        if (output_id != target_id) {
            correct = false;
        }
        if (i + 1 < target_ids.size()) {
            ret = runtime.eval_logits(model_id, target_id, logits);
            if (ret != rwkvmobile::RWKV_SUCCESS || logits.data_ptr == nullptr) {
                return ret ? ret : rwkvmobile::RWKV_ERROR_RUNTIME;
            }
        }
    }
    output_text = runtime.tokenizer_decode(model_id, output_ids);
    return rwkvmobile::RWKV_SUCCESS;
}

int main(int argc, char **argv) {
    std::cout.setf(std::ios::unitbuf);
    if (argc < 5 || argc > 6) {
        std::cerr << "Usage: " << argv[0] << " <tokenizer_path> <model_path> <backend> <text_path> [--decode-only]\n";
        return 1;
    }

    std::string tokenizer_path = argv[1];
    std::string model_path = argv[2];
    std::string backend = argv[3];
    std::string text_path = argv[4];
    const bool decode_only = (argc == 6 && std::string(argv[5]) == "--decode-only");
    if (argc == 6 && !decode_only) {
        std::cerr << "Unknown argument: " << argv[5] << "\n";
        return 1;
    }

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
        int ret = decode_only
            ? run_evaluation_decode_only(runtime, model_id, prompt, target, correct, logits_val, output_text, true)
            : runtime.run_evaluation(model_id, prompt, target, correct, logits_val, output_text, true);
        ENSURE_SUCCESS_OR_LOG_EXIT(ret, "Evaluation failed");
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
