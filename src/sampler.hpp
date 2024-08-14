#ifndef SAMPLER_HPP
#define SAMPLER_HPP

#include <random>
#include <algorithm>
#include <vector>

class Sampler {
public:
    Sampler() {
        _generator.seed(std::random_device()());
    }

    int Sample(const float* logits, const size_t size, float temperature, int top_k, float top_p) {
        temperature = std::clamp(temperature, 0.1f, 5.f);
        if (top_k >= size)
            top_k = size;

        if (top_k == 0 || top_k == 1)
            return std::max_element(logits, logits + size) - logits;

        // softmax
        float sum = 0;
        int *index = new int[size];
        float *probs = new float[size];

        const float max_logit = *std::max_element(logits, logits + size);

        for (int i = 0; i < size; i++) {
            probs[i] = std::exp(logits[i] - max_logit);
            sum += probs[i];
            index[i] = i;
        }

        if (top_k != size)
            std::nth_element(index, index + top_k,
                    index + size,
                    [&](int i, int j) { return probs[i] > probs[j]; });
        std::sort(index, index + top_k,
                [&](int i, int j) { return probs[i] > probs[j]; });

        int len = top_k;

        // top-p
        float cumsum = 0;
        for (int i = 0; i < len; i++) {
            probs[index[i]] /= sum;
            cumsum += probs[index[i]];
            if (cumsum >= top_p) {
                len = i + 1;
                break;
            }
        }

        // temperature
        if (fabs(temperature - 1.f) > 1e-6) {
            cumsum = 0;
            for (int i = 0; i < len; i++) {
                probs[index[i]] = std::pow(probs[index[i]], 1.f / temperature);
                cumsum += probs[index[i]];
            }
        }

        // random choice
        float random_value = 1. * (_generator() - _generator.min()) /
                            (_generator.max() - _generator.min()) * cumsum;
        
        int ret = -1;
        cumsum = 0;
        for (int i = 0; i < len; i++) {
            cumsum += probs[index[i]];
            if (cumsum >= random_value) {
                ret = index[i];
                break;
            }
        }
        
        delete[] index;
        delete[] probs;
        return ret;
    }

    void set_seed(int seed) {
        _generator.seed(seed);
    }
private:
    std::minstd_rand0 _generator;
};

#endif