#ifndef RUNTIME_H
#define RUNTIME_H

#include <string>
#include <map>
#include "backend.h"
#include "tokenizer.h"
#include "sampler.h"

namespace rwkvmobile {

class runtime {
public:
    runtime() {};
    ~runtime() {};
    int init(std::string backend_name);
    int init(int backend_id);
    int load_model(std::string model_path);
    int load_tokenizer(std::string vocab_file);
    int eval_logits(int id, std::vector<float> &logits);
    int eval_logits(std::vector<int> ids, std::vector<float> &logits);
    int get_state(std::vector<float> &state);
    int set_state(std::vector<float> state);
    int release();

    int chat(std::string user_role, std::string response_role, std::string user_input, std::string &response);
    int gen_completion(std::string user_input, std::string &response, int max_length);

    int set_seed(int64_t seed);
    inline void set_sampler_params(float temperature, int top_k, float top_p) {
        _temperature = temperature;
        _top_k = top_k;
        _top_p = top_p;
    }

    inline void set_penalty_params(float presence_penalty, float frequency_penalty, float penalty_decay) {
        _presence_penalty = presence_penalty;
        _frequency_penalty = frequency_penalty;
        _penalty_decay = penalty_decay;
    }

    inline float get_temperature() { return _temperature; }
    inline int get_top_k() { return _top_k; }
    inline float get_top_p() { return _top_p; }
    inline float get_presence_penalty() { return _presence_penalty; }
    inline float get_frequency_penalty() { return _frequency_penalty; }
    inline float get_penalty_decay() { return _penalty_decay; }

    std::string get_available_backends_str();
    int get_available_backend_ids(std::vector<int> &backend_ids);
    std::string backend_id_to_str(int backend_id) {
        return backend_enum_to_str(backend_id);
    }
    int backend_str_to_id(std::string backend_str) {
        return backend_str_to_enum(backend_str);
    }

private:
    std::unique_ptr<execution_provider> _backend;
    std::unique_ptr<tokenizer_base> _tokenizer;
    std::unique_ptr<sampler> _sampler;

    int _vocab_size = 65536;

    float _temperature = 1.0;
    int _top_k = 128;
    float _top_p = 0.3;
    float _presence_penalty = 0.0;
    float _frequency_penalty = 1.0;
    float _penalty_decay = 0.996;

    std::map<int, float> _occurences;
};

}

#endif