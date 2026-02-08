#include "sampler.h"
#include <mutex>
#include <cmath>
#include <limits>

namespace rwkvmobile {

NucleusSampler::NucleusSampler() {
    _seed = std::random_device()();
    _generator.seed(_seed);

    _temperature = std::vector<float>(_max_batch_size, 1.0f);
    _top_k = std::vector<int>(_max_batch_size, 128);
    _top_p = std::vector<float>(_max_batch_size, 0.5f);
    _presence_penalty = std::vector<float>(_max_batch_size, 2.0f);
    _frequency_penalty = std::vector<float>(_max_batch_size, 0.2f);
    _penalty_decay = std::vector<float>(_max_batch_size, 0.996f);
}

int NucleusSampler::sample(const Tensor1D & logits, const size_t size) {
    return sample(logits, size, _temperature[0], _top_k[0], _top_p[0], _index_buffer, _probs_buffer);
}

int NucleusSampler::sample(const Tensor1D & logits, const size_t size, float temperature, int top_k, float top_p) {
    return sample(logits, size, temperature, top_k, top_p, _index_buffer, _probs_buffer);
}

int NucleusSampler::sample(const Tensor1D & logits, const size_t size, float temperature, int top_k, float top_p, std::vector<int> &index_buffer, std::vector<float> &probs_buffer) {
    if (!logits.data_ptr) {
        return 0;
    }
    temperature = std::clamp(temperature, 0.1f, 5.f);
    if (size == 0) return 0;
    if (top_k <= 0 || (size_t)top_k > size) top_k = (int)size;

    if (top_k == 1 || std::fabs(top_p - 0.f) < 1e-4f) {
        if (logits.dtype == TensorDType::F16) {
            return std::max_element(static_cast<const half_float::half*>(logits.data_ptr), static_cast<const half_float::half*>(logits.data_ptr) + size) - static_cast<const half_float::half*>(logits.data_ptr);
        } else if (logits.dtype == TensorDType::F32) {
            return std::max_element(static_cast<const float*>(logits.data_ptr), static_cast<const float*>(logits.data_ptr) + size) - static_cast<const float*>(logits.data_ptr);
        } else {
            // TODO
            return 0;
        }
    }

    // Keep only top-k indices using a min-heap (convert only scalars as needed).
    struct Item { float v; int i; };
    std::vector<Item> heap;
    heap.reserve((size_t)top_k);
    for (size_t i = 0; i < size; ++i) {
        const float v = tensor1d_get_f32(logits, i);
        if ((int)heap.size() < top_k) {
            heap.push_back({v, (int)i});
            if ((int)heap.size() == top_k) {
                std::make_heap(heap.begin(), heap.end(), [](const Item& a, const Item& b){ return a.v > b.v; }); // min-heap
            }
        } else if (v > heap.front().v) {
            std::pop_heap(heap.begin(), heap.end(), [](const Item& a, const Item& b){ return a.v > b.v; });
            heap.back() = {v, (int)i};
            std::push_heap(heap.begin(), heap.end(), [](const Item& a, const Item& b){ return a.v > b.v; });
        }
    }
    std::sort(heap.begin(), heap.end(), [](const Item& a, const Item& b){ return a.v > b.v; }); // desc by logit

    const int len0 = (int)heap.size();
    if ((int)index_buffer.size() < len0) index_buffer.resize(len0);
    if ((int)probs_buffer.size() < len0) probs_buffer.resize(len0);
    for (int k = 0; k < len0; ++k) {
        index_buffer[k] = heap[k].i;
        probs_buffer[k] = heap[k].v; // store logits temporarily
    }

    // softmax on top-k only
    float sum = 0.0f;
    const float max_logit = probs_buffer[0];
    for (int k = 0; k < len0; ++k) {
        probs_buffer[k] = std::exp((probs_buffer[k] - max_logit) / temperature);
        sum += probs_buffer[k];
    }

    // top-p
    float cumsum = 0.0f;
    int len = len0;
    for (int k = 0; k < len0; ++k) {
        probs_buffer[k] /= sum;
        cumsum += probs_buffer[k];
        if (cumsum >= top_p) {
            len = k + 1;
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

std::vector<int> NucleusSampler::sample_batch(const Tensor1D & logits, const size_t sampling_size, const size_t hstep, int batch_size) {
    return sample_batch(logits, sampling_size, hstep, batch_size, _temperature, _top_k, _top_p);
}

std::vector<int> NucleusSampler::sample_batch(const Tensor1D & logits, const size_t sampling_size, const size_t hstep, int batch_size, std::vector<float> temperature, std::vector<int> top_k, std::vector<float> top_p) {
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
        Tensor1D view = tensor1d_subview(logits, (size_t)i * hstep, sampling_size);
        ret[i] = sample(view, sampling_size, temperature[i], top_k[i], top_p[i], _batch_index_buffer[i], _batch_probs_buffer[i]);
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

void NucleusSampler::apply_penalties(Tensor1D & logits, const size_t size, std::map<int, float> &occurences, std::vector<int> token_banned, float presence_penalty, float frequency_penalty, float penalty_decay) {
    if (!logits.data_ptr) {
        return;
    }
    for (auto &[id, occurence] : occurences) {
        if (id >= size) {
            continue;
        }
        tensor1d_add_bias(logits, (size_t)id, -(frequency_penalty * occurence + presence_penalty));
        occurences[id] *= penalty_decay;
    }

    for (auto &token : token_banned) {
        if (token >= size) {
            continue;
        }
        tensor1d_set_f32(logits, (size_t)token, -std::numeric_limits<float>::infinity());
    }
}

void NucleusSampler::apply_penalties(Tensor1D & logits, const size_t size) {
    if (_presence_penalty[0] > 0.0f && _frequency_penalty[0] > 0.0f && _penalty_decay[0] > 0.0f) {
        apply_penalties(logits, size, _occurences, _token_banned, _presence_penalty[0], _frequency_penalty[0], _penalty_decay[0]);
    }
}

std::vector<int> NucleusSampler::sample_topk_greedy(const Tensor1D & logits, const size_t size, int top_k) {
    if (!logits.data_ptr) {
        return std::vector<int>();
    }
    if (size == 0) return std::vector<int>();
    if (top_k <= 0 || (size_t)top_k > size) top_k = (int)size;

    // Keep only top-k indices using a min-heap (convert only scalars as needed).
    struct Item { float v; int i; };
    std::vector<Item> heap;
    heap.reserve((size_t)top_k);
    for (size_t i = 0; i < size; ++i) {
        const float v = tensor1d_get_f32(logits, i);
        if ((int)heap.size() < top_k) {
            heap.push_back({v, (int)i});
            if ((int)heap.size() == top_k) {
                std::make_heap(heap.begin(), heap.end(), [](const Item& a, const Item& b){ return a.v > b.v; }); // min-heap
            }
        } else if (v > heap.front().v) {
            std::pop_heap(heap.begin(), heap.end(), [](const Item& a, const Item& b){ return a.v > b.v; });
            heap.back() = {v, (int)i};
            std::push_heap(heap.begin(), heap.end(), [](const Item& a, const Item& b){ return a.v > b.v; });
        }
    }
    std::sort(heap.begin(), heap.end(), [](const Item& a, const Item& b){ return a.v > b.v; }); // desc by logit

    std::vector<int> ret;
    ret.reserve((size_t)top_k);
    for (int i = 0; i < top_k; i++) {
        ret.push_back(heap[i].i);
    }
    return ret;
}

}