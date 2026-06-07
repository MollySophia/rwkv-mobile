#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "commondef.h"
#include "runtime.h"

#define ENSURE_SUCCESS_OR_LOG_EXIT(x, msg) if ((x) != rwkvmobile::RWKV_SUCCESS) { std::cout << msg << std::endl; return 1; }

namespace {

std::string read_text_file(const char *path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return {};
    }
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

int copy_logits_to_f32(const rwkvmobile::Tensor1D &logits, std::vector<float> &out, int vocab_size) {
    if (logits.data_ptr == nullptr || logits.count < (size_t)vocab_size) {
        return rwkvmobile::RWKV_ERROR_RUNTIME;
    }
    out.resize((size_t)vocab_size);
    if (logits.dtype == rwkvmobile::TensorDType::F32) {
        std::copy_n(reinterpret_cast<const float *>(logits.data_ptr), vocab_size, out.data());
        return rwkvmobile::RWKV_SUCCESS;
    }
    if (logits.dtype == rwkvmobile::TensorDType::F16) {
        const half_float::half *h = reinterpret_cast<const half_float::half *>(logits.data_ptr);
        for (int i = 0; i < vocab_size; ++i) {
            out[(size_t)i] = (float)h[i];
        }
        return rwkvmobile::RWKV_SUCCESS;
    }
    return rwkvmobile::RWKV_ERROR_UNSUPPORTED;
}

double cross_entropy_loss(const std::vector<float> &logits, int target_id) {
    const float max_logit = *std::max_element(logits.begin(), logits.end());
    double sum = 0.0;
    for (float v : logits) {
        sum += std::exp((double)v - (double)max_logit);
    }
    return std::log(sum) + (double)max_logit - (double)logits[(size_t)target_id];
}

double elapsed_seconds(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

std::string safe_basename(const char *path) {
    std::string name(path);
    const size_t slash = name.find_last_of("/\\");
    if (slash != std::string::npos) {
        name = name.substr(slash + 1);
    }
    for (char &c : name) {
        const bool ok = (c >= 'a' && c <= 'z') ||
                        (c >= 'A' && c <= 'Z') ||
                        (c >= '0' && c <= '9') ||
                        c == '.' || c == '_' || c == '-';
        if (!ok) {
            c = '_';
        }
    }
    return name;
}

} // namespace

int main(int argc, char **argv) {
    std::cout.setf(std::ios::unitbuf);
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <tokenizer_path> <model_path> <backend> <text_path> [text_path...]\n";
        return 1;
    }

    rwkvmobile::Runtime runtime;
    int model_id = runtime.load_model(argv[2], argv[3], argv[1], nullptr);
    ENSURE_SUCCESS_OR_LOG_EXIT(model_id < 0 ? model_id : rwkvmobile::RWKV_SUCCESS, "Failed to load model");
    if (model_id < 0) return 1;

    const int vocab_size = runtime.get_vocab_size(model_id);
    std::vector<float> logits_f32;
    const char *csv_prefix = std::getenv("RWKV_LONG_CONTEXT_CSV_PREFIX");

    double decode_avg_loss_weighted_sum = 0.0;
    int evaluated_files = 0;
    int64_t decode_loss_tokens_total = 0;

    for (int file_idx = 4; file_idx < argc; ++file_idx) {
        const char *text_path = argv[file_idx];
        std::string text = read_text_file(text_path);
        if (text.empty()) {
            std::cerr << "Failed to read or empty file: " << text_path << std::endl;
            runtime.release();
            return 1;
        }

        std::vector<int> ids = runtime.tokenizer_encode(model_id, text);
        if (ids.size() > 8193) {
            ids.resize(8193);
        }
        if (ids.size() < 2) {
            std::cerr << "Too few tokens in file: " << text_path << std::endl;
            runtime.release();
            return 1;
        }

        std::cout << "File: " << text_path << std::endl;
        std::cout << "Tokens used: " << ids.size() << std::endl;

        std::ofstream csv;
        if (csv_prefix != nullptr && csv_prefix[0] != '\0') {
            const std::string csv_path = std::string(csv_prefix) + "_" + safe_basename(text_path) + ".csv";
            csv.open(csv_path, std::ios::binary);
            if (!csv) {
                std::cerr << "Failed to open CSV: " << csv_path << std::endl;
                runtime.release();
                return 1;
            }
            csv.precision(10);
            csv << "token_count,loss,cumulative_loss,cumulative_avg_loss,cumulative_ppl,target_id\n";
            std::cout << "CSV: " << csv_path << std::endl;
        }

        rwkvmobile::Tensor1D logits;
        runtime.clear_state(model_id);
        runtime.reset_inference_speed_stats(model_id);
        double decode_loss_sum = 0.0;
        auto start = std::chrono::steady_clock::now();
        for (size_t i = 0; i + 1 < ids.size(); ++i) {
            int ret = runtime.eval_logits(model_id, ids[i], logits);
            ENSURE_SUCCESS_OR_LOG_EXIT(ret, "Decode eval failed");
            ENSURE_SUCCESS_OR_LOG_EXIT(copy_logits_to_f32(logits, logits_f32, vocab_size), "Decode logits copy failed");
            const double loss = cross_entropy_loss(logits_f32, ids[i + 1]);
            decode_loss_sum += loss;
            if (csv) {
                const size_t token_count = i + 1;
                const double cumulative_avg_loss = decode_loss_sum / (double)token_count;
                csv << token_count << ","
                    << loss << ","
                    << decode_loss_sum << ","
                    << cumulative_avg_loss << ","
                    << std::exp(cumulative_avg_loss) << ","
                    << ids[i + 1] << "\n";
            }
            if ((i + 1) % 1024 == 0) {
                std::cout << "  decode progress: " << (i + 1) << "/" << (ids.size() - 1) << std::endl;
            }
        }
        const double decode_seconds = elapsed_seconds(start);
        const double decode_avg_loss = decode_loss_sum / (double)(ids.size() - 1);
        const double decode_tok_s = runtime.get_avg_decode_speed(model_id);

        std::cout << "Decode avg next-token loss: " << decode_avg_loss
                  << ", ppl=" << std::exp(decode_avg_loss)
                  << ", seconds=" << decode_seconds
                  << ", tok/s=" << decode_tok_s << std::endl;
        std::cout << "====================================" << std::endl;

        decode_avg_loss_weighted_sum += decode_loss_sum;
        decode_loss_tokens_total += (int64_t)ids.size() - 1;
        evaluated_files++;
    }

    if (evaluated_files > 0 && decode_loss_tokens_total > 0) {
        const double mean_decode_avg = decode_avg_loss_weighted_sum / (double)decode_loss_tokens_total;
        std::cout << "Summary files: " << evaluated_files << std::endl;
        std::cout << "Summary decode tokens: " << decode_loss_tokens_total << std::endl;
        std::cout << "Weighted decode avg next-token loss: " << mean_decode_avg
                  << ", ppl=" << std::exp(mean_decode_avg) << std::endl;
    }

    runtime.release_model(model_id);
    runtime.release();
    return 0;
}
