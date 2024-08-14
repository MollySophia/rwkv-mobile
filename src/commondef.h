#ifndef COMMONDEF_H
#define COMMONDEF_H

#include <string>

namespace rwkvmobile {

enum {
    RWKV_SUCCESS = 0,
    RWKV_ERROR_IO = 1 << 0,
    RWKV_ERROR_INIT = 1 << 1,
    RWKV_ERROR_EVAL = 1 << 2,
    RWKV_ERROR_INVALID_PARAMETERS = 1 << 3,
    RWKV_ERROR_BACKEND = 1 << 4,
    RWKV_ERROR_MODEL = 1 << 5,
    RWKV_ERROR_TOKENIZER = 1 << 6,
    RWKV_ERROR_SAMPLER = 1 << 7,
};

enum {
    RWKV_BACKEND_RWKVCPP = 0,
    RWKV_BACKEND_COUNT,
};

std::string backend_enum_to_str(int backend) {
    switch (backend) {
        case RWKV_BACKEND_RWKVCPP:
            return "rwkv.cpp";
        default:
            return "unknown";
    }
}

int backend_str_to_enum(std::string backend) {
    if (backend == "rwkv.cpp") {
        return RWKV_BACKEND_RWKVCPP;
    }
    return -1;
}

} // namespace rwkvmobile

#endif