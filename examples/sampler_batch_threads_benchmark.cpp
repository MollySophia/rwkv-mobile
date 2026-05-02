#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "sampler.h"

namespace {

struct ExplicitSamplerBuffers {
    std::vector<std::vector<int>> index_buffers;
    std::vector<std::vector<float>> probs_buffers;
};

#ifdef ANDROID
constexpr int kActualSamplerThreadCap = 4;
#else
constexpr int kActualSamplerThreadCap = 8;
#endif

std::vector<int> parse_int_list(const std::string &text) {
    std::vector<int> values;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            values.push_back(std::atoi(item.c_str()));
        }
    }
    values.erase(std::remove_if(values.begin(), values.end(), [](int v) { return v <= 0; }), values.end());
    return values;
}

void print_usage(const char *argv0) {
    std::cout
        << "Usage: " << argv0 << " [options]\n"
        << "Options:\n"
        << "  --vocab <n>          Vocabulary size, default 65536\n"
        << "  --iters <n>          Timed sample_batch calls per case, default 200\n"
        << "  --warmup <n>         Warmup sample_batch calls per case, default 20\n"
        << "  --top-k <n>          top_k, default 128\n"
        << "  --top-p <float>      top_p, default 0.5\n"
        << "  --temperature <f>    temperature, default 1.0\n"
        << "  --batches <list>     Comma separated batch sizes, default 1,2,4,8,16,32\n"
        << "  --threads <list>     Comma separated OpenMP thread counts, default up to CPU cores\n"
        << "  --repeat <n>         Timed repeats per case; report median/best, default 3\n"
        << "  --mode <mode>        actual, explicit, or both, default both\n";
}

std::vector<int> sample_batch_with_explicit_threads(
    rwkvmobile::NucleusSampler &sampler,
    const rwkvmobile::Tensor1D &logits,
    size_t sampling_size,
    size_t hstep,
    int batch_size,
    const std::vector<float> &temperatures,
    const std::vector<int> &top_ks,
    const std::vector<float> &top_ps,
    int threads,
    ExplicitSamplerBuffers &buffers) {
    std::vector<int> ret(batch_size);
    if ((int)buffers.index_buffers.size() < batch_size) {
        buffers.index_buffers.resize(batch_size);
    }
    if ((int)buffers.probs_buffers.size() < batch_size) {
        buffers.probs_buffers.resize(batch_size);
    }

#ifdef _OPENMP
    #pragma omp parallel for num_threads(threads)
#endif
    for (int i = 0; i < batch_size; i++) {
        rwkvmobile::Tensor1D view = rwkvmobile::tensor1d_subview(logits, (size_t)i * hstep, sampling_size);
        ret[i] = sampler.sample(view, sampling_size, temperatures[i], top_ks[i], top_ps[i],
            buffers.index_buffers[i], buffers.probs_buffers[i]);
    }
    return ret;
}

} // namespace

int main(int argc, char **argv) {
    int vocab_size = 65536;
    int iterations = 200;
    int warmup = 20;
    int top_k = 128;
    float top_p = 0.5f;
    float temperature = 1.0f;
    int repeat = 3;
    std::vector<int> batch_sizes = {1, 2, 4, 8, 16, 32};
    std::vector<int> requested_threads;
    std::string mode = "both";

    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        auto need_value = [&](const char *name) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << std::endl;
                std::exit(1);
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--vocab") {
            vocab_size = std::atoi(need_value("--vocab"));
        } else if (arg == "--iters") {
            iterations = std::atoi(need_value("--iters"));
        } else if (arg == "--warmup") {
            warmup = std::atoi(need_value("--warmup"));
        } else if (arg == "--top-k") {
            top_k = std::atoi(need_value("--top-k"));
        } else if (arg == "--top-p") {
            top_p = std::atof(need_value("--top-p"));
        } else if (arg == "--temperature") {
            temperature = std::atof(need_value("--temperature"));
        } else if (arg == "--batches") {
            batch_sizes = parse_int_list(need_value("--batches"));
        } else if (arg == "--threads") {
            requested_threads = parse_int_list(need_value("--threads"));
        } else if (arg == "--repeat") {
            repeat = std::atoi(need_value("--repeat"));
        } else if (arg == "--mode") {
            mode = need_value("--mode");
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    if (vocab_size <= 0 || iterations <= 0 || warmup < 0 || repeat <= 0 || top_k <= 0 || batch_sizes.empty()) {
        std::cerr << "Invalid benchmark parameters" << std::endl;
        return 1;
    }
    if (mode != "actual" && mode != "explicit" && mode != "both") {
        std::cerr << "Invalid --mode: " << mode << std::endl;
        return 1;
    }

#ifdef _OPENMP
    omp_set_dynamic(0);
    const int cpu_threads = std::max(1, omp_get_num_procs());
    if (requested_threads.empty()) {
        for (int t : {1, 2, 4, 6, 8, 12, 16, 18, 24, 32}) {
            if (t <= cpu_threads) {
                requested_threads.push_back(t);
            }
        }
        if (requested_threads.empty() || requested_threads.back() != cpu_threads) {
            requested_threads.push_back(cpu_threads);
        }
    }
#else
    const int cpu_threads = 1;
    requested_threads = {1};
#endif

    const int max_batch_size = *std::max_element(batch_sizes.begin(), batch_sizes.end());
    std::vector<float> logits_storage((size_t)max_batch_size * (size_t)vocab_size);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 3.0f);
    for (float &v : logits_storage) {
        v = dist(rng);
    }

    std::cout << "# openmp=";
#ifdef _OPENMP
    std::cout << "on";
#else
    std::cout << "off";
#endif
    std::cout << ", cpu_threads=" << cpu_threads
              << ", actual_thread_cap=" << kActualSamplerThreadCap << "\n";
    std::cout << "mode,batch_size,requested_threads,effective_threads,vocab_size,top_k,top_p,temperature,iterations,repeat,median_ms,best_ms,median_us_per_step,median_us_per_slot,median_slots_per_sec,checksum\n";

    long long global_checksum = 0;
    for (int batch_size : batch_sizes) {
        rwkvmobile::Tensor1D logits = rwkvmobile::Tensor1D::make(
            logits_storage.data(), rwkvmobile::TensorDType::F32, (size_t)batch_size * (size_t)vocab_size);
        std::vector<float> temperatures(batch_size, temperature);
        std::vector<int> top_ks(batch_size, top_k);
        std::vector<float> top_ps(batch_size, top_p);

        for (int threads : requested_threads) {
            for (const std::string &case_mode : {std::string("actual"), std::string("explicit")}) {
                if (mode != "both" && mode != case_mode) {
                    continue;
                }

#ifdef _OPENMP
                omp_set_num_threads(threads);
                const int effective_threads = case_mode == "actual"
                    ? std::max(1, std::min({batch_size, kActualSamplerThreadCap, threads}))
                    : std::max(1, threads);
#else
                const int effective_threads = 1;
#endif
                rwkvmobile::NucleusSampler sampler;
                sampler.set_seed(42);
                ExplicitSamplerBuffers explicit_buffers;

                auto run_once = [&]() {
                    if (case_mode == "actual") {
                        return sampler.sample_batch(logits, (size_t)vocab_size, (size_t)vocab_size,
                            batch_size, temperatures, top_ks, top_ps);
                    }
                    return sample_batch_with_explicit_threads(sampler, logits, (size_t)vocab_size,
                        (size_t)vocab_size, batch_size, temperatures, top_ks, top_ps,
                        effective_threads, explicit_buffers);
                };

                long long checksum = 0;
                for (int i = 0; i < warmup; i++) {
                    auto ids = run_once();
                    checksum += ids.empty() ? 0 : ids[(size_t)i % ids.size()];
                }

                std::vector<double> elapsed_ms_samples;
                elapsed_ms_samples.reserve((size_t)repeat);
                for (int r = 0; r < repeat; r++) {
                    auto start = std::chrono::steady_clock::now();
                    for (int i = 0; i < iterations; i++) {
                        auto ids = run_once();
                        checksum += ids.empty() ? 0 : ids[(size_t)i % ids.size()];
                    }
                    auto end = std::chrono::steady_clock::now();
                    elapsed_ms_samples.push_back(std::chrono::duration<double, std::milli>(end - start).count());
                }

                std::sort(elapsed_ms_samples.begin(), elapsed_ms_samples.end());
                const double median_ms = elapsed_ms_samples[elapsed_ms_samples.size() / 2];
                const double best_ms = elapsed_ms_samples.front();
                const double us_per_step = median_ms * 1000.0 / iterations;
                const double us_per_slot = us_per_step / batch_size;
                const double slots_per_sec = (double)iterations * batch_size * 1000.0 / median_ms;
                global_checksum += checksum;

                std::cout
                    << case_mode << ','
                    << batch_size << ','
                    << threads << ','
                    << effective_threads << ','
                    << vocab_size << ','
                    << top_k << ','
                    << top_p << ','
                    << temperature << ','
                    << iterations << ','
                    << repeat << ','
                    << std::fixed << std::setprecision(3) << median_ms << ','
                    << best_ms << ','
                    << us_per_step << ','
                    << us_per_slot << ','
                    << slots_per_sec << ','
                    << checksum << '\n';
            }
        }
    }

    std::cerr << "checksum=" << global_checksum << std::endl;
    return 0;
}
