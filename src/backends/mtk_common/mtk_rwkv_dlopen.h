#ifndef MTK_RWKV_DLOPEN_H
#define MTK_RWKV_DLOPEN_H

#include <string>

namespace rwkvmobile {

struct MtkRwkvApi {
    void* set_log_callback = nullptr;
    void* init = nullptr;
    void* release = nullptr;
    void* inference_once = nullptr;
    void* inference_batch = nullptr;
    void* prefill = nullptr;
    void* eval_with_embeddings = nullptr;
    void* reset = nullptr;
    void* get_att_state_size = nullptr;
    void* get_wkv_state_size = nullptr;
    void* get_ffn_state_size = nullptr;
    void* get_att_state = nullptr;
    void* get_wkv_state = nullptr;
    void* get_ffn_state = nullptr;
    void* set_att_state = nullptr;
    void* set_wkv_state = nullptr;
    void* set_ffn_state = nullptr;
    void* get_att_state_slot = nullptr;
    void* get_wkv_state_slot = nullptr;
    void* get_ffn_state_slot = nullptr;
    void* set_att_state_slot = nullptr;
    void* set_wkv_state_slot = nullptr;
    void* set_ffn_state_slot = nullptr;
    void* zero_state_slot = nullptr;
};

class MtkRwkvDlopen {
public:
    MtkRwkvDlopen() = default;
    ~MtkRwkvDlopen();

    MtkRwkvDlopen(const MtkRwkvDlopen&) = delete;
    MtkRwkvDlopen& operator=(const MtkRwkvDlopen&) = delete;

    int open(const char* tag, const char* env_var, const char* default_library_name, void* extra);
    void close();

    bool is_loaded() const { return _handle != nullptr; }
    const MtkRwkvApi& api() const { return _api; }
    MtkRwkvApi& api() { return _api; }
    const std::string& path() const { return _path; }

private:
    void* _handle = nullptr;
    MtkRwkvApi _api{};
    std::string _tag;
    std::string _path;
};

} // namespace rwkvmobile

#endif
