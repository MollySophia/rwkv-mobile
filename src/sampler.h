#ifndef SAMPLER_HPP
#define SAMPLER_HPP

#include <random>
#include <algorithm>
#include <vector>
#include <map>
#include <mutex>

#include "tensor.h"

namespace rwkvmobile {

class NucleusSampler {
public:
    NucleusSampler();

    void apply_penalties(Tensor1D & logits, const size_t size, std::map<int, float> &occurences, std::vector<int> token_banned, float presence_penalty, float frequency_penalty, float penalty_decay);

    void apply_penalties(Tensor1D & logits, const size_t size);

    void clear_occurences() { _occurences.clear(); }

    void update_occurences(int token);

    int sample(const Tensor1D & logits, const size_t size);

    int sample(const Tensor1D & logits, const size_t size, float temperature, int top_k, float top_p);

    int sample(const Tensor1D & logits, const size_t size, float temperature, int top_k, float top_p, std::vector<int> &index_buffer, std::vector<float> &probs_buffer);

    std::vector<int> sample_batch(const Tensor1D & logits, const size_t sampling_size, const size_t hstep, int batch_size);

    std::vector<int> sample_batch(const Tensor1D & logits, const size_t sampling_size, const size_t hstep, int batch_size, std::vector<float> temperature, std::vector<int> top_k, std::vector<float> top_p);

    std::vector<int> sample_topk_greedy(const Tensor1D & logits, const size_t size, int top_k);

    void set_seed(int32_t seed);
    int get_seed();

    void set_temperature(float temperature) { _temperature = std::vector<float>(_max_batch_size, temperature)   ; }
    void set_top_k(int top_k) { _top_k = std::vector<int>(_max_batch_size, top_k); }
    void set_top_p(float top_p) { _top_p = std::vector<float>(_max_batch_size, top_p); }
    void set_presence_penalty(float presence_penalty) { _presence_penalty = std::vector<float>(_max_batch_size, presence_penalty); }
    void set_frequency_penalty(float frequency_penalty) { _frequency_penalty = std::vector<float>(_max_batch_size, frequency_penalty); }
    void set_penalty_decay(float penalty_decay) { _penalty_decay = std::vector<float>(_max_batch_size, penalty_decay); }
    void set_token_banned(std::vector<int> token_banned) { _token_banned = token_banned; }

    void set_temperature_on_batch_slot(int slot, float temperature) { if (slot >= 0 && slot < _max_batch_size) _temperature[slot] = temperature; }
    void set_top_k_on_batch_slot(int slot, int top_k) { if (slot >= 0 && slot < _max_batch_size) _top_k[slot] = top_k; }
    void set_top_p_on_batch_slot(int slot, float top_p) { if (slot >= 0 && slot < _max_batch_size) _top_p[slot] = top_p; }
    void set_presence_penalty_on_batch_slot(int slot, float presence_penalty) { if (slot >= 0 && slot < _max_batch_size) _presence_penalty[slot] = presence_penalty; }
    void set_frequency_penalty_on_batch_slot(int slot, float frequency_penalty) { if (slot >= 0 && slot < _max_batch_size) _frequency_penalty[slot] = frequency_penalty; }
    void set_penalty_decay_on_batch_slot(int slot, float penalty_decay) { if (slot >= 0 && slot < _max_batch_size) _penalty_decay[slot] = penalty_decay; }

    float get_temperature() { return _temperature[0]; }
    int get_top_k() { return _top_k[0]; }
    float get_top_p() { return _top_p[0]; }
    float get_presence_penalty() { return _presence_penalty[0]; }
    float get_frequency_penalty() { return _frequency_penalty[0]; }
    float get_penalty_decay() { return _penalty_decay[0]; }
    std::vector<int> get_token_banned() { return _token_banned; }

    float get_temperature_on_batch_slot(int slot) { return _temperature[std::max(0, std::min(slot, _max_batch_size - 1))]; }
    int get_top_k_on_batch_slot(int slot) { return _top_k[std::max(0, std::min(slot, _max_batch_size - 1))]; }
    float get_top_p_on_batch_slot(int slot) { return _top_p[std::max(0, std::min(slot, _max_batch_size - 1))]; }
    float get_presence_penalty_on_batch_slot(int slot) { return _presence_penalty[std::max(0, std::min(slot, _max_batch_size - 1))]; }
    float get_frequency_penalty_on_batch_slot(int slot) { return _frequency_penalty[std::max(0, std::min(slot, _max_batch_size - 1))]; }
    float get_penalty_decay_on_batch_slot(int slot) { return _penalty_decay[std::max(0, std::min(slot, _max_batch_size - 1))]; }

private:
    std::mutex _mutex;
    std::minstd_rand0 _generator;

    std::vector<float> _probs_buffer;
    std::vector<int> _index_buffer;

    std::vector<std::vector<float>> _batch_probs_buffer;
    std::vector<std::vector<int>> _batch_index_buffer;

    std::vector<int> _token_banned;

    int _max_batch_size = 32;

    std::vector<float> _temperature;
    std::vector<int> _top_k;
    std::vector<float> _top_p;
    std::vector<float> _presence_penalty;
    std::vector<float> _frequency_penalty;
    std::vector<float> _penalty_decay;

    int32_t _seed = 42;

    std::map<int, float> _occurences;
};

}
#endif