#ifndef WEB_RWKV_BACKEND_H
#define WEB_RWKV_BACKEND_H

#include "backend.h"
#include "web_rwkv_ffi.h"
#include <atomic>
#include <memory>
#include <mutex>

namespace rwkvmobile {

class web_rwkv_state {
public:
    web_rwkv_state(StateRaw raw) : raw(raw) {
    }

    ~web_rwkv_state() {
        if (raw.state && raw.len > 0) {
            ::free_state(raw);
        }
    }


    web_rwkv_state(const web_rwkv_state&) = delete;
    web_rwkv_state& operator=(const web_rwkv_state&) = delete;

    web_rwkv_state(web_rwkv_state&& other) noexcept
        : raw(other.raw) {
        other.raw.state = nullptr;
        other.raw.len = 0;
    }

    web_rwkv_state& operator=(web_rwkv_state&& other) noexcept {
        if (this != &other) {
            if (raw.state && raw.len > 0) {
                ::free_state(raw);
            }
            raw = other.raw;
            other.raw.state = nullptr;
            other.raw.len = 0;
        }
        return *this;
    }

    StateRaw raw;
};

class web_rwkv_backend : public execution_provider {
public:
    ~web_rwkv_backend() {
        release_model();
        release();
    }
    int init(void * extra) override;
    int load_model(std::string model_path, void * extra = nullptr) override;
    float get_load_progress() const override;
    int eval(int id, Tensor1D & logits) override;
    int eval(std::vector<int> ids, Tensor1D & logits) override;
    int eval_batch(std::vector<std::vector<int>> ids, Tensor1D & logits) override;

    bool is_available() override;
    int zero_state() override;
    int get_state(std::any &state) override;
    int set_state(std::any state) override;
    int free_state(std::any state) override;

    int get_state_on_batch_slot(int slot, std::any &state) override;
    int set_state_on_batch_slot(int slot, std::any state) override;
    int zero_state_on_batch_slot(int slot) override;

    int serialize_runtime_state(std::any state, std::vector<uint8_t> &data) override;
    int deserialize_runtime_state(std::vector<uint8_t> &data, std::any &state) override;

    int release_model() override;
    int release() override;

    void set_load_progress_real(float p) { _load_progress_real = p; }
private:
    std::vector<float> logits_buffer;

    std::atomic<bool> _load_is_pth{false};
    std::atomic<float> _load_progress_real{-1.f};
    mutable std::mutex _load_progress_mutex;
    mutable float _load_progress_reported = 0.f;
    mutable float _load_progress_step = 0.1f;  // >=0.5 时使用的步长，从 0.1 逐渐减小
};

}

#endif
