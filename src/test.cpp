#include "sampler_old.hpp"
#include "sampler.hpp"
#include <vector>
#include <iostream>
#include <chrono>
#include <random>

int main() {
    float *probs = new float[65536];
    long time_total = 0, time_total_old = 0;

    Sampler_old sampler_old;
    Sampler sampler;

    srand(time(NULL));

    const int num_samples = 1000;

    for (int j = 0; j < 65536; j++) {
        probs[j] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX)) * 20;
    }

    int correct = 0;
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < 65536; j++) {
            probs[j] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX)) * 20;
        }

        int seed = time(NULL);
        sampler.set_seed(seed);
        sampler_old.set_seed(seed);

        auto start = std::chrono::high_resolution_clock::now();
        int id = sampler.Sample(probs, 65536, 1.0, 65536, 1.0);
        auto end = std::chrono::high_resolution_clock::now();
        time_total += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        int id_old = sampler_old.Sample(probs, 65536, 1.0, 65536, 1.0);
        end = std::chrono::high_resolution_clock::now();
        time_total_old += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        if (id == id_old) {
            correct++;
        }
    }

    std::cout << "New sampler top_k=65536 time: " << time_total / 1000000.0 / num_samples << " ms" << std::endl;
    std::cout << "Old sampler top_k=65536 time: " << time_total_old / 1000000.0 / num_samples << " ms" << std::endl;
    std::cout << "Correct: " << correct << "/" << num_samples << std::endl;

    time_total = 0;
    time_total_old = 0;
    correct = 0;
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < 65536; j++) {
            probs[j] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX)) * 20;
        }

        int seed = time(NULL);
        sampler.set_seed(seed);
        sampler_old.set_seed(seed);

        auto start = std::chrono::high_resolution_clock::now();
        int id = sampler.Sample(probs, 65536, 1.0, 128, 1.0);
        auto end = std::chrono::high_resolution_clock::now();
        time_total += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        int id_old = sampler_old.Sample(probs, 65536, 1.0, 128, 1.0);
        end = std::chrono::high_resolution_clock::now();
        time_total_old += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        if (id == id_old) {
            correct++;
        }
    }

    std::cout << "New sampler top_k=128 time: " << time_total / 1000000.0 / num_samples << " ms" << std::endl;
    std::cout << "Old sampler top_k=128 time: " << time_total_old / 1000000.0 / num_samples << " ms" << std::endl;
    std::cout << "Correct: " << correct << "/" << num_samples << std::endl;

    time_total = 0;
    time_total_old = 0;
    correct = 0;
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < 65536; j++) {
            probs[j] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX)) * 20;
        }

        int seed = time(NULL);
        sampler.set_seed(seed);
        sampler_old.set_seed(seed);

        auto start = std::chrono::high_resolution_clock::now();
        int id = sampler.Sample(probs, 65536, 1.0, 8, 1.0);
        auto end = std::chrono::high_resolution_clock::now();
        time_total += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        int id_old = sampler_old.Sample(probs, 65536, 1.0, 8, 1.0);
        end = std::chrono::high_resolution_clock::now();
        time_total_old += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        if (id == id_old) {
            correct++;
        }
    }

    std::cout << "New sampler top_k=8 time: " << time_total / 1000000.0 / num_samples << " ms" << std::endl;
    std::cout << "Old sampler top_k=8 time: " << time_total_old / 1000000.0 / num_samples << " ms" << std::endl;
    std::cout << "Correct: " << correct << "/" << num_samples << std::endl;
}
