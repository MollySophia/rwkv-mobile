#ifndef SAMPLER_HPP
#define SAMPLER_HPP

#include <random>
#include <algorithm>
#include <vector>

namespace rwkvmobile {

class sampler {
public:
    sampler();

    int sample(const float* logits, const size_t size, float temperature, int top_k, float top_p);

    void set_seed(int seed);
private:
    std::minstd_rand0 _generator;
};

}
#endif