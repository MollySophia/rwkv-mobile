#include <fstream>

#include "backend.h"
#include "web_rwkv_ffi.h"
#include "web_rwkv_backend.h"
#include "commondef.h"

namespace rwkvmobile {

#ifdef ENABLE_WEBRWKV
int web_rwkv_backend::init(void * extra) {
    web_rwkv_init((uint64_t)time(NULL));
    return RWKV_SUCCESS;
}

int web_rwkv_backend::load_model(std::string model_path) {
    if (!std::filesystem::exists(model_path)) {
        return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
    }
    web_rwkv_load(model_path.c_str(), 999, 999);
    return RWKV_SUCCESS;
}

int web_rwkv_backend::eval(int id, std::vector<float> &logits) {
    std::vector<uint16_t> ids = {(uint16_t)id};
    int ret = web_rwkv_infer_logits(ids.data(), ids.size(), logits.data(), logits.size());
    if (!ret) {
        return RWKV_SUCCESS;
    } else {
        return RWKV_ERROR_EVAL;
    }
}

int web_rwkv_backend::eval(std::vector<int> ids, std::vector<float> &logits) {
    std::vector<uint16_t> ids_u16(ids.begin(), ids.end());
    int ret = web_rwkv_infer_logits(ids_u16.data(), ids_u16.size(), logits.data(), logits.size());
    if (!ret) {
        return RWKV_SUCCESS;
    } else {
        return RWKV_ERROR_EVAL;
    }
}

bool web_rwkv_backend::is_available() {
    // TODO: Detect this
    return true;
}

#else

int web_rwkv_backend::init(void * extra) {
    return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
}

int web_rwkv_backend::load_model(std::string model_path) {
    return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
}

int web_rwkv_backend::eval(int id, std::vector<float> &logits) {
    return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
}

int web_rwkv_backend::eval(std::vector<int> ids, std::vector<float> &logits) {
    return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
}

bool web_rwkv_backend::is_available() {
    return false;
}

#endif

} // namespace rwkvmobile