#include <iostream>
#include <chrono>
#include "../src/multimodal/vision/vision_encoder.h"

int main(int argc, char **argv) {
    // set stdout to be unbuffered
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <model> <adapter> <image>" << std::endl;
        return 1;
    }

    rwkvmobile::VisionEncoder * encoder = new rwkvmobile::VisionEncoder();
    encoder->load_model(argv[1], argv[2]);

    double total_time = 0;
    int n_runs = 10;
    for (int i = 0; i < n_runs; i++) {

        auto start = std::chrono::high_resolution_clock::now();
        std::vector<float> embeddings;
        int n_tokens;
        encoder->encode(argv[3], embeddings, n_tokens);
        auto end = std::chrono::high_resolution_clock::now();
        total_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }
    std::cout << "Average time: " << total_time / n_runs << " ms" << std::endl;

    return 0;
}
