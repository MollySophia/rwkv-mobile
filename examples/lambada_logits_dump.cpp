#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "commondef.h"
#include "runtime.h"
#include "tensor.h"

#define ENSURE_SUCCESS_OR_LOG_EXIT(x, msg) if ((x) != rwkvmobile::RWKV_SUCCESS) { std::cerr << msg << std::endl; return 1; }

namespace {

std::vector<std::string> read_lambada_texts(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open text file: " + path);
    }
    const size_t file_size = (size_t)file.tellg();
    std::string buf(file_size, '\0');
    file.seekg(0, std::ios::beg);
    file.read(buf.data(), (std::streamsize)file_size);

    std::vector<std::string> texts;
    size_t next = 0;
    for (size_t i = 0; i < buf.size(); ++i) {
        if (buf[i] == '|') {
            texts.emplace_back(buf.data() + next, i - next);
            next = i + 1;
        }
    }
    return texts;
}

std::set<int> parse_indices(const std::string& csv) {
    std::set<int> out;
    std::stringstream ss(csv);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            out.insert(std::atoi(item.c_str()));
        }
    }
    return out;
}

std::vector<float> logits_to_f32(const rwkvmobile::Tensor1D& logits, int vocab) {
    std::vector<float> out((size_t)vocab);
    if (logits.data_ptr == nullptr || logits.count < (size_t)vocab) {
        throw std::runtime_error("invalid logits tensor");
    }
    if (logits.dtype == rwkvmobile::TensorDType::F32) {
        std::copy_n(reinterpret_cast<const float*>(logits.data_ptr), vocab, out.data());
    } else if (logits.dtype == rwkvmobile::TensorDType::F16) {
        const auto* h = reinterpret_cast<const half_float::half*>(logits.data_ptr);
        for (int i = 0; i < vocab; ++i) {
            out[(size_t)i] = (float)h[i];
        }
    } else {
        throw std::runtime_error("unsupported logits dtype");
    }
    return out;
}

struct LogitStats {
    int argmax = -1;
    float max_logit = 0.0f;
    float min_logit = 0.0f;
    float mean = 0.0f;
    float stddev = 0.0f;
    float logsumexp = 0.0f;
    float entropy = 0.0f;
};

LogitStats compute_stats(const std::vector<float>& logits) {
    LogitStats st;
    st.argmax = (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
    st.max_logit = logits[(size_t)st.argmax];
    st.min_logit = *std::min_element(logits.begin(), logits.end());

    double sum = 0.0;
    for (float v : logits) sum += v;
    st.mean = (float)(sum / (double)logits.size());

    double var = 0.0;
    for (float v : logits) {
        const double d = (double)v - st.mean;
        var += d * d;
    }
    st.stddev = (float)std::sqrt(var / (double)logits.size());

    double exp_sum = 0.0;
    for (float v : logits) exp_sum += std::exp((double)v - st.max_logit);
    st.logsumexp = st.max_logit + (float)std::log(exp_sum);

    double entropy = 0.0;
    for (float v : logits) {
        const double p = std::exp((double)v - st.logsumexp);
        if (p > 0.0) {
            entropy -= p * std::log(p);
        }
    }
    st.entropy = (float)entropy;
    return st;
}

std::vector<int> top_indices(const std::vector<float>& logits, int k) {
    std::vector<int> indices(logits.size());
    std::iota(indices.begin(), indices.end(), 0);
    if (k < (int)indices.size()) {
        std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
            [&](int a, int b) { return logits[(size_t)a] > logits[(size_t)b]; });
        indices.resize((size_t)k);
    } else {
        std::sort(indices.begin(), indices.end(),
            [&](int a, int b) { return logits[(size_t)a] > logits[(size_t)b]; });
    }
    return indices;
}

void dump_step(
    rwkvmobile::Runtime& runtime,
    int model_id,
    const rwkvmobile::Tensor1D& logits,
    int target_id,
    int sample_idx,
    int target_pos,
    int top_k) {

    const int vocab = (int)logits.count;
    std::vector<float> f32 = logits_to_f32(logits, vocab);
    const LogitStats st = compute_stats(f32);
    const float target_logit = f32[(size_t)target_id];
    const float logprob = target_logit - st.logsumexp;
    const float prob = std::exp(logprob);
    const auto top = top_indices(f32, top_k);

    std::cout << std::setprecision(9);
    std::cout << "STEP sample=" << sample_idx
              << " target_pos=" << target_pos
              << " target_id=" << target_id
              << " target_text=\"" << runtime.tokenizer_decode(model_id, target_id) << "\""
              << " argmax_id=" << st.argmax
              << " argmax_text=\"" << runtime.tokenizer_decode(model_id, st.argmax) << "\""
              << " correct=" << (st.argmax == target_id ? 1 : 0)
              << " target_logit=" << target_logit
              << " target_logprob=" << logprob
              << " target_prob=" << prob
              << " nll=" << -logprob
              << " max_logit=" << st.max_logit
              << " min_logit=" << st.min_logit
              << " mean=" << st.mean
              << " std=" << st.stddev
              << " logsumexp=" << st.logsumexp
              << " entropy=" << st.entropy
              << " dtype=" << (logits.dtype == rwkvmobile::TensorDType::F16 ? "f16" : "f32")
              << std::endl;

    std::cout << "TOP sample=" << sample_idx << " target_pos=" << target_pos;
    for (int id : top) {
        const float lp = f32[(size_t)id] - st.logsumexp;
        std::cout << " [" << id
                  << ",\"" << runtime.tokenizer_decode(model_id, id) << "\""
                  << ",logit=" << f32[(size_t)id]
                  << ",prob=" << std::exp(lp)
                  << "]";
    }
    std::cout << std::endl;
}

} // namespace

int main(int argc, char **argv) {
    std::cout.setf(std::ios::unitbuf);
    if (argc < 5 || argc > 7) {
        std::cerr << "Usage: " << argv[0]
                  << " <tokenizer_path> <model_path> <backend> <text_path> [sample_indices_csv] [top_k]\n";
        return 1;
    }

    const std::string tokenizer_path = argv[1];
    const std::string model_path = argv[2];
    const std::string backend = argv[3];
    const std::string text_path = argv[4];
    const std::set<int> wanted = argc >= 6 ? parse_indices(argv[5]) : parse_indices("0,4,5,7,9,34,37");
    const int top_k = argc >= 7 ? std::max(1, std::atoi(argv[6])) : 10;

    rwkvmobile::Runtime runtime;
    int model_id = runtime.load_model(model_path, backend, tokenizer_path, nullptr);
    ENSURE_SUCCESS_OR_LOG_EXIT(model_id < 0 ? model_id : rwkvmobile::RWKV_SUCCESS, "Failed to load model");

    std::vector<std::string> eval_text;
    try {
        eval_text = read_lambada_texts(text_path);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    std::cout << "Eval texts num: " << eval_text.size() << std::endl;
    for (int sample_idx = 0; sample_idx < (int)eval_text.size(); ++sample_idx) {
        if (!wanted.empty() && wanted.find(sample_idx) == wanted.end()) {
            continue;
        }

        const auto& text = eval_text[(size_t)sample_idx];
        const size_t split = text.find_last_of(' ');
        if (split == std::string::npos) {
            std::cerr << "Skipping sample without space: " << sample_idx << std::endl;
            continue;
        }
        const std::string prompt = text.substr(0, split);
        const std::string target = text.substr(split);
        std::vector<int> source_ids = runtime.tokenizer_encode(model_id, prompt);
        std::vector<int> target_ids = runtime.tokenizer_encode(model_id, target);
        source_ids.insert(source_ids.begin(), 0);

        std::cout << "SAMPLE index=" << sample_idx
                  << " prompt_tokens=" << source_ids.size()
                  << " target_tokens=" << target_ids.size()
                  << " prompt=\"" << prompt << "\""
                  << " target=\"" << target << "\""
                  << std::endl;
        std::cout << "TARGET_IDS index=" << sample_idx;
        for (int id : target_ids) {
            std::cout << " " << id << "(\"" << runtime.tokenizer_decode(model_id, id) << "\")";
        }
        std::cout << std::endl;

        runtime.clear_state(model_id);
        rwkvmobile::Tensor1D logits;
        int ret = runtime.eval_logits(model_id, source_ids, logits);
        if (ret || logits.data_ptr == nullptr) {
            std::cerr << "eval_logits failed on prefill for sample " << sample_idx << std::endl;
            return 1;
        }

        double sample_logprob = 0.0;
        bool correct = true;
        for (int pos = 0; pos < (int)target_ids.size(); ++pos) {
            const int target_id = target_ids[(size_t)pos];
            std::vector<float> f32 = logits_to_f32(logits, (int)logits.count);
            const LogitStats st = compute_stats(f32);
            sample_logprob += (double)f32[(size_t)target_id] - st.logsumexp;
            correct = correct && (st.argmax == target_id);
            dump_step(runtime, model_id, logits, target_id, sample_idx, pos, top_k);
            if (pos + 1 < (int)target_ids.size()) {
                ret = runtime.eval_logits(model_id, target_id, logits);
                if (ret || logits.data_ptr == nullptr) {
                    std::cerr << "eval_logits failed on decode for sample " << sample_idx << std::endl;
                    return 1;
                }
            }
        }
        std::cout << "SAMPLE_SUMMARY index=" << sample_idx
                  << " correct=" << (correct ? 1 : 0)
                  << " logprob=" << sample_logprob
                  << " nll=" << -sample_logprob
                  << " ppl=" << std::exp(-sample_logprob / std::max<size_t>(1, target_ids.size()))
                  << std::endl;
    }

    runtime.release_model(model_id);
    runtime.release();
    return 0;
}
