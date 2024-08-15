#ifndef BACKEND_H
#define BACKEND_H

#include <string>
#include <vector>

#include "commondef.h"

namespace rwkvmobile {

class execution_provider {
public:
    virtual int init(void * extra) { return 0; }
    virtual int init(std::string model_path, void * extra) { return 0; }
    virtual int load_model(std::string model_path) { return RWKV_ERROR_MODEL; }
    virtual int eval(int id, std::vector<float> &logits) { return 0; };
    virtual int eval(std::vector<int> ids, std::vector<float> &logits) { return 0; };
    virtual int get_state(std::vector<float> &state) { return 0; }
    virtual int set_state(std::vector<float> state) { return 0; }
    virtual int release_model() { return 0; };
    virtual int release() { return 0; };
    virtual bool is_available() { return false; };
};

enum {
    RWKV_BACKEND_RWKVCPP = 0,
    RWKV_BACKEND_WEBRWKV,
    RWKV_BACKEND_COUNT,
};

std::string backend_enum_to_str(int backend);
int backend_str_to_enum(std::string backend);

}

#endif