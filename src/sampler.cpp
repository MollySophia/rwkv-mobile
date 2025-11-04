#include "sampler.h"
#include <mutex>

namespace rwkvmobile {

NucleusSampler::NucleusSampler() {
    _seed = std::random_device()();
    _generator.seed(_seed);
}

int NucleusSampler::sample(const float* logits, const size_t size) {
    return sample(logits, size, _temperature, _top_k, _top_p, _index_buffer, _probs_buffer);
}

int NucleusSampler::sample(const float* logits, const size_t size, float temperature, int top_k, float top_p) {
    return sample(logits, size, temperature, top_k, top_p, _index_buffer, _probs_buffer);
}

int NucleusSampler::sample(const float* logits, const size_t size, float temperature, int top_k, float top_p, std::vector<int> &index_buffer, std::vector<float> &probs_buffer) {
    if (logits == nullptr) {
        return 0;
    }
    temperature = std::clamp(temperature, 0.1f, 5.f);
    if (top_k >= size || top_k == 0)
        top_k = size;

    if (top_k == 1 || fabs(top_p - 0.f) < 1e-4)
        return std::max_element(logits, logits + size) - logits;

    if (index_buffer.size() != size) {
        index_buffer.resize(size);
    }
    for (int i = 0; i < size; i++) {
        index_buffer[i] = i;
    }

    if (top_k != size)
        std::nth_element(index_buffer.begin(), index_buffer.begin() + top_k,
                index_buffer.end(),
                [&](int i, int j) { return logits[i] > logits[j]; });
    std::sort(index_buffer.begin(), index_buffer.begin() + top_k,
            [&](int i, int j) { return logits[i] > logits[j]; });

    int len = top_k;
    if (probs_buffer.size() < len) {
        probs_buffer.resize(len);
    }

    // softmax
    float sum = 0;
    for (int i = 0; i < len; i++) {
        probs_buffer[i] = std::exp((logits[index_buffer[i]] - logits[index_buffer[0]]) / temperature);
        sum += probs_buffer[i];
    }

    // top-p
    float cumsum = 0;
    for (int i = 0; i < len; i++) {
        probs_buffer[i] /= sum;
        cumsum += probs_buffer[i];
        if (cumsum >= top_p) {
            len = i + 1;
            break;
        }
    }

    // random choice
    float random_value;
    {
        std::lock_guard<std::mutex> lock(_mutex);
        random_value = cumsum * (_generator() - _generator.min()) / (_generator.max() - _generator.min());
    }

    int ret = index_buffer[0];
    cumsum = 0;
    for (int i = 0; i < len; i++) {
        cumsum += probs_buffer[i];
        if (cumsum >= random_value) {
            ret = index_buffer[i];
            break;
        }
    }
    return ret;
}

std::vector<int> NucleusSampler::sample_batch(const float* logits, const size_t sampling_size, const size_t hstep, int batch_size) {
    return sample_batch(logits, sampling_size, hstep, batch_size, std::vector<float>(batch_size, _temperature), std::vector<int>(batch_size, _top_k), std::vector<float>(batch_size, _top_p));
}

std::vector<int> NucleusSampler::sample_batch(const float* logits, const size_t sampling_size, const size_t hstep, int batch_size, std::vector<float> temperature, std::vector<int> top_k, std::vector<float> top_p) {
    std::vector<int> ret(batch_size);

    if (temperature.size() == 1) {
        temperature = std::vector<float>(batch_size, temperature[0]);
    }

    if (top_k.size() == 1) {
        top_k = std::vector<int>(batch_size, top_k[0]);
    }

    if (top_p.size() == 1) {
        top_p = std::vector<float>(batch_size, top_p[0]);
    }

    if (_batch_index_buffer.size() < batch_size) {
        _batch_index_buffer.resize(batch_size);
    }
    if (_batch_probs_buffer.size() < batch_size) {
        _batch_probs_buffer.resize(batch_size);
    }

    #pragma omp parallel for
    for (int i = 0; i < batch_size; i++) {
        ret[i] = sample(logits + i * hstep, sampling_size, temperature[i], top_k[i], top_p[i], _batch_index_buffer[i], _batch_probs_buffer[i]);
    }
    return ret;
}

void NucleusSampler::set_seed(int32_t seed) {
    _seed = seed;
    _generator.seed(_seed);
}

int NucleusSampler::get_seed() {
    return _seed;
}

void NucleusSampler::update_occurences(int token) {
    if (_occurences.find(token) == _occurences.end()) {
        _occurences[token] = 0;
    }
    _occurences[token] += 1.0f;
}

void NucleusSampler::apply_penalties(float * logits, const size_t size, std::map<int, float> &occurences, std::vector<int> token_banned, float presence_penalty, float frequency_penalty, float penalty_decay) {
    if (!logits) {
        return;
    }
    for (auto &[id, occurence] : occurences) {
        if (id >= size) {
            continue;
        }
        logits[id] -=
            frequency_penalty * occurence + presence_penalty;
        occurences[id] *= penalty_decay;
    }

    for (auto &token : token_banned) {
        if (token >= size) {
            continue;
        }
        logits[token] = -INFINITY;
    }
}

void NucleusSampler::apply_penalties(float * logits, const size_t size) {
    if (_presence_penalty > 0.0f && _frequency_penalty > 0.0f && _penalty_decay > 0.0f) {
        apply_penalties(logits, size, _occurences, _token_banned, _presence_penalty, _frequency_penalty, _penalty_decay);
    }
}

}