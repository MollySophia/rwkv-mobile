#ifndef COMMONDEF_H
#define COMMONDEF_H

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
    RWKV_ERROR_RUNTIME = 1 << 8,
    RWKV_ERROR_UNSUPPORTED = 1 << 9,
};

} // namespace rwkvmobile

#endif