#include <iostream>
#include <chrono>
#include <algorithm>
#include <cstdlib>
#include "../src/multimodal/vision/vision_encoder.h"

int main(int argc, char **argv) {
    // set stdout to be unbuffered
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc != 4 && argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <model> <adapter-or-> <image> [n_runs]" << std::endl;
        return 1;
    }

    rwkvmobile::VisionEncoder * encoder = new rwkvmobile::VisionEncoder();
    std::string adapter_path = argv[2];
    if (adapter_path == "-") {
        adapter_path.clear();
    }
    int ret = encoder->load_model(argv[1], adapter_path);
    if (ret != 0) {
        std::cerr << "Failed to load vision encoder, ret=" << ret << std::endl;
        delete encoder;
        return 1;
    }

    double total_time = 0;
    int n_runs = argc == 5 ? std::max(1, std::atoi(argv[4])) : 10;
    for (int i = 0; i < n_runs; i++) {

        auto start = std::chrono::high_resolution_clock::now();
        std::vector<float> embeddings;
        int n_tokens;
        if (!encoder->encode(argv[3], embeddings, n_tokens)) {
            std::cerr << "Failed to encode image" << std::endl;
            delete encoder;
            return 1;
        }
        auto end = std::chrono::high_resolution_clock::now();
        total_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }
    std::cout << "Average time: " << total_time / n_runs << " ms" << std::endl;

    delete encoder;
    return 0;
}
