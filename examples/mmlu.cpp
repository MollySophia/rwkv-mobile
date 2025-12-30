#include <cstdio>
#include <ctime>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>

#include "commondef.h"
#include "runtime.h"
#include "json.hpp"

using json = nlohmann::json;

struct MMLU_Question {
    std::string prompt;
    std::string answer;
    std::string subject;
};

struct scoreboard {
    int total;
    int correct;
};

#define ENSURE_SUCCESS_OR_LOG_EXIT(x, msg) if (x != rwkvmobile::RWKV_SUCCESS) { std::cout << msg << std::endl; return 1; }

int main(int argc, char ** argv) {
    // number of tokens to predict
    int n_predict = 1;

    std::cout.setf(std::ios::unitbuf);
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <tokenizer_path> <model_path> <backend> <text_path>\n";
        return 1;
    }

    std::string tokenizer_path = argv[1];
    std::string model_path = argv[2];
    std::string backend = argv[3];
    std::string text_path = argv[4];

    // load dataset
    std::ifstream file(text_path);
    if (!file.is_open()) {
        throw std::runtime_error("can not load dataset");
    }
    std::string prompt_template = "User: You are a very talented expert in <SUBJECT>. Answer this question:\n<Q>\nA. <|A|>\nB. <|B|>\nC. <|C|>\nD. <|D|>\n\nAssistant: The answer is";

    auto replace_all = [](std::string& str, const std::string& from, const std::string& to) {
        size_t pos = 0;
        while((pos = str.find(from)) != std::string::npos) {
            str.replace(pos, from.length(), to);
        }
    };

    json data = json::parse(file);
    std::vector<MMLU_Question> questions;
    for (const auto& item : data) {
        std::string prompt = prompt_template;
        std::string subject = item["subject"].get<std::string>();
        replace_all(subject, "_", " ");
        replace_all(prompt, "<SUBJECT>", subject);
        replace_all(prompt, "<Q>", item["question"].get<std::string>());
        replace_all(prompt, "<|A|>", item["choices"][0].get<std::string>());
        replace_all(prompt, "<|B|>", item["choices"][1].get<std::string>());
        replace_all(prompt, "<|C|>", item["choices"][2].get<std::string>());
        replace_all(prompt, "<|D|>", item["choices"][3].get<std::string>());
        MMLU_Question q;
        q.prompt = prompt;
        q.answer = " " + item["answer"].get<std::string>();
        q.subject = subject;
        questions.push_back(q);
    }

    rwkvmobile::Runtime runtime;
    int model_id = runtime.load_model(model_path, backend, tokenizer_path, nullptr);
    ENSURE_SUCCESS_OR_LOG_EXIT(model_id < 0 ? model_id : rwkvmobile::RWKV_SUCCESS, "Failed to load model");
    if (model_id < 0) return 1;
    runtime.set_sampler_params(model_id, 1.0, 1, 1.0);
    runtime.set_penalty_params(model_id, 0.0, 0.0, 0.0);

    rwkvmobile::Tensor1D output;
    auto softmax_fp32 = [](const float *logits, size_t size) {
        std::vector<float> probs(size);
        float max_val = *std::max_element(logits, logits + size);
        float sum = 0;
        for (size_t i = 0; i < size; i++) {
            probs[i] = std::exp((logits[i] - max_val));
            sum += probs[i];
        }
        for (size_t i = 0; i < size; i++) {
            probs[i] /= sum;
        }
        return probs;
    };

    auto softmax_fp16_to_fp32 = [](const half_float::half *logits, size_t size) {
        std::vector<float> probs(size);
        half_float::half max_val = *std::max_element(logits, logits + size);
        float sum = 0;
        for (size_t i = 0; i < size; i++) {
            probs[i] = std::exp((float)logits[i] - max_val);
            sum += probs[i];
        }
        for (size_t i = 0; i < size; i++) {
            probs[i] /= sum;
        }
        return probs;
    };

    std::map<std::string, int> choices_tokens = {
        {" A", runtime.tokenizer_encode(model_id, " A")[0]},
        {" B", runtime.tokenizer_encode(model_id, " B")[0]},
        {" C", runtime.tokenizer_encode(model_id, " C")[0]},
        {" D", runtime.tokenizer_encode(model_id, " D")[0]},
    };

    // main loop
    int total_correct = 0;
    std::map<std::string, scoreboard> score_by_subject;
    for (size_t i = 0; i < questions.size(); i++) {
    
        const auto q = questions[i];

        std::string prompt = q.prompt;
        auto prompt_tokens = runtime.tokenizer_encode(model_id, prompt);
        prompt_tokens.insert(prompt_tokens.begin(), 0);

        std::string answer;
        runtime.clear_state(model_id);
        runtime.eval_logits(model_id, prompt_tokens, output);
        std::vector<float> probs;
        const float* logits_ptr = nullptr;
        if (output.dtype == rwkvmobile::TensorDType::F32) {
            probs = softmax_fp32(reinterpret_cast<const float*>(output.data_ptr), output.count);
        } else if (output.dtype == rwkvmobile::TensorDType::F16) {
            probs = softmax_fp16_to_fp32(reinterpret_cast<const half_float::half*>(output.data_ptr), output.count);
        } else {
            std::cerr << "Unsupported logits dtype in mmlu.cpp" << std::endl;
            return 1;
        }

        // answer = runtime.tokenizer_decode(model_id, output_id);
        auto max_prob = 0.0f;
        for (const auto& choice : choices_tokens) {
            if (probs[choice.second] > max_prob) {
                max_prob = probs[choice.second];
                answer = choice.first;
            }
        }

        score_by_subject[q.subject].total++;
        if (answer == q.answer) {
            total_correct++;
            score_by_subject[q.subject].correct++;
        }
        if (i % 10 == 0) { printf("%lu/%lu, correct: %d, acc: %f\n", i+1, questions.size(), total_correct, (float)total_correct/(i+1)); }
    }

    printf("\ncorrect: %d, acc: %f\n", total_correct, (float)total_correct/questions.size());

    json json_output;
    json_output["model"] = model_path;
    json_output["total"] = questions.size();
    json_output["correct"] = total_correct;
    json_output["total_accuracy"] = static_cast<float>(total_correct) / questions.size();
    for (const auto& score : score_by_subject) {
        json_output[score.first] = {
            {"total", score.second.total},
            {"correct", score.second.correct},
            {"accuracy", static_cast<float>(score.second.correct) / score.second.total}
        };
    }

    time_t rawtime;
    struct tm *info;
    char buffer[80];
    time( &rawtime );
    info = localtime( &rawtime );
    strftime(buffer, 80, "%Y-%m-%d %H:%M:%S", info);
    std::string file_name = "mmlu_results_" + std::string(buffer) + ".json";

    std::ofstream out_file(file_name);
    out_file << json_output.dump(2) << std::endl;

    runtime.release_model(model_id);
    runtime.release();

    return 0;
}
