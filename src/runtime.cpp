#include "runtime.h"
#include "backend.h"
#include "web_rwkv_backend.h"

namespace rwkvmobile {

std::string backend_enum_to_str(int backend) {
    switch (backend) {
        case RWKV_BACKEND_RWKVCPP:
            return "rwkv.cpp";
        case RWKV_BACKEND_WEBRWKV:
            return "web-rwkv";
        default:
            return "unknown";
    }
}

int backend_str_to_enum(std::string backend) {
    if (backend == "rwkv.cpp") {
        return RWKV_BACKEND_RWKVCPP;
    } else if (backend == "web-rwkv") {
        return RWKV_BACKEND_WEBRWKV;
    }
    return -1;
}

int runtime::init(std::string backend_name) {
    int backend_id = backend_str_to_enum(backend_name);
    if (backend_id < 0) {
        return RWKV_ERROR_BACKEND;
    }
    return init(backend_id);
}

int runtime::init(int backend_id) {
    _sampler = std::unique_ptr<sampler>(new sampler);
    if (_sampler == nullptr) {
        return RWKV_ERROR_SAMPLER;
    }

    if (backend_id == RWKV_BACKEND_WEBRWKV) {
        _backend = std::unique_ptr<execution_provider>(new web_rwkv_backend);
    } else {
        return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
    }
    return _backend->init(nullptr);
}

int runtime::load_model(std::string model_path) {
    if (_backend == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    return _backend->load_model(model_path);
}

int runtime::load_tokenizer(std::string vocab_file) {
    if (_tokenizer != nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    _tokenizer = std::unique_ptr<tokenizer_base>(new trie_tokenizer);
    if (_tokenizer == nullptr) {
        return RWKV_ERROR_TOKENIZER;
    }
    return _tokenizer->load(vocab_file);
}

int runtime::eval_logits(int id, std::vector<float> &logits) {
    if (_backend == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    return _backend->eval(id, logits);
}

int runtime::eval_logits(std::vector<int> ids, std::vector<float> &logits) {
    if (_backend == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    return _backend->eval(ids, logits);
}

int runtime::chat(std::string user_role, std::string response_role, std::string user_input, std::string &response, const int max_length) {
    if (_backend == nullptr || _tokenizer == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    std::string prompt = user_role + ": " + user_input + "\n\n" + response_role + ":";
    std::vector<int> ids = _tokenizer->encode(prompt);
    std::vector<float> logits(_vocab_size);
    response = "";
    int ret = eval_logits(ids, logits);
    if (ret) {
        return ret;
    }

    for (int i = 0; i < max_length; i++) {
        for (auto &[id, occurence] : _occurences) {
            logits[id] -=
                _frequency_penalty * occurence + _presence_penalty;
            occurence *= _penalty_decay;
        }

        int idx = _sampler->sample(logits.data(), logits.size(), _temperature, _top_k, _top_p);
        if (idx == 0) {
            break;
        }
        _occurences[idx]++;

        response += _tokenizer->decode(idx);
        ret = eval_logits(idx, logits);
        if (response.c_str()[response.size() - 1] == '\n' && response.c_str()[response.size() - 2] == '\n') {
            break;
        }
        if (ret) {
            return ret;
        }
    }

    return RWKV_SUCCESS;
}

int runtime::gen_completion(std::string prompt, std::string &completion, int length) {
    if (_backend == nullptr || _tokenizer == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    std::vector<int> ids = _tokenizer->encode(prompt);
    std::vector<float> logits(_vocab_size);
    completion = "";
    int ret = eval_logits(ids, logits);
    if (ret) {
        return ret;
    }

    for (int i = 0; i < length; i++) {
        for (auto &[id, occurence] : _occurences) {
            logits[id] -=
                _frequency_penalty * occurence + _presence_penalty;
            occurence *= _penalty_decay;
        }

        int idx = _sampler->sample(logits.data(), logits.size(), _temperature, _top_k, _top_p);
        if (idx == 0) {
            break;
        }
        _occurences[idx]++;

        completion += _tokenizer->decode(idx);
        ret = eval_logits(idx, logits);
        if (ret) {
            return ret;
        }
    }

    return RWKV_SUCCESS;
}

} // namespace rwkvmobile
