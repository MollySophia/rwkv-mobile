#ifndef RWKVMOBILE_TENSOR_H
#define RWKVMOBILE_TENSOR_H

#ifdef max
#undef max
#endif

#ifdef min
#undef min
#endif

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "half.hpp"

namespace rwkvmobile {

enum class TensorDType : uint8_t {
    UNKNOWN = 0,
    F16,
    F32,
    I16,
    I32,
    U8,
    I8,
};

inline constexpr size_t tensor_dtype_bytes(TensorDType dt) {
    switch (dt) {
        case TensorDType::F16: return 2;
        case TensorDType::F32: return 4;
        case TensorDType::I16: return 2;
        case TensorDType::I32: return 4;
        case TensorDType::U8:  return 1;
        case TensorDType::I8:  return 1;
        default:               return 0;
    }
}

// A minimal 1D tensor that can be either a non-owning view or an owning copy.
// When _owned_storage is non-empty, data_ptr points into it and the tensor
// owns its memory.  Otherwise it is a lightweight view (the old behaviour).
struct Tensor1D {
    void* data_ptr = nullptr;
    TensorDType dtype = TensorDType::UNKNOWN;
    size_t count = 0;

    size_t bytes = 0;
    size_t bytes_per_elem = 0;

    float scale = 1.0f;
    float offset = 0.0f;

    std::vector<uint8_t> _owned_storage;

    Tensor1D() = default;
    ~Tensor1D() = default;

    Tensor1D(const Tensor1D& o)
        : dtype(o.dtype), count(o.count), bytes(o.bytes),
          bytes_per_elem(o.bytes_per_elem), scale(o.scale), offset(o.offset),
          _owned_storage(o._owned_storage)
    {
        data_ptr = _owned_storage.empty() ? o.data_ptr : _owned_storage.data();
    }

    Tensor1D(Tensor1D&& o) noexcept
        : data_ptr(o.data_ptr), dtype(o.dtype), count(o.count),
          bytes(o.bytes), bytes_per_elem(o.bytes_per_elem),
          scale(o.scale), offset(o.offset),
          _owned_storage(std::move(o._owned_storage))
    {
        if (!_owned_storage.empty()) {
            data_ptr = _owned_storage.data();
        }
        o.data_ptr = nullptr;
        o.count = 0;
        o.bytes = 0;
    }

    Tensor1D& operator=(const Tensor1D& o) {
        if (this != &o) {
            dtype = o.dtype;
            count = o.count;
            bytes = o.bytes;
            bytes_per_elem = o.bytes_per_elem;
            scale = o.scale;
            offset = o.offset;
            _owned_storage = o._owned_storage;
            data_ptr = _owned_storage.empty() ? o.data_ptr : _owned_storage.data();
        }
        return *this;
    }

    Tensor1D& operator=(Tensor1D&& o) noexcept {
        if (this != &o) {
            dtype = o.dtype;
            count = o.count;
            bytes = o.bytes;
            bytes_per_elem = o.bytes_per_elem;
            scale = o.scale;
            offset = o.offset;
            _owned_storage = std::move(o._owned_storage);
            data_ptr = _owned_storage.empty() ? o.data_ptr : _owned_storage.data();
            o.data_ptr = nullptr;
            o.count = 0;
            o.bytes = 0;
        }
        return *this;
    }

    // Create a non-owning view (same as before).
    static inline Tensor1D make(void* ptr, TensorDType dt, size_t n) {
        Tensor1D t;
        t.data_ptr = ptr;
        t.dtype = dt;
        t.count = n;
        t.bytes_per_elem = tensor_dtype_bytes(dt);
        t.bytes = t.bytes_per_elem * t.count;
        return t;
    }

    // Deep-copy the data into a new owning Tensor1D.
    Tensor1D copy() const {
        Tensor1D t;
        t.dtype = dtype;
        t.count = count;
        t.bytes_per_elem = bytes_per_elem;
        t.bytes = bytes;
        t.scale = scale;
        t.offset = offset;
        if (data_ptr && bytes > 0) {
            t._owned_storage.resize(bytes);
            std::memcpy(t._owned_storage.data(), data_ptr, bytes);
            t.data_ptr = t._owned_storage.data();
        }
        return t;
    }

    bool is_view() const { return _owned_storage.empty() && data_ptr != nullptr; }
    bool is_owned() const { return !_owned_storage.empty(); }

    inline bool valid() const { return data_ptr != nullptr && count > 0 && bytes_per_elem > 0; }
};

// Always returns a non-owning view into t's data (never copies _owned_storage).
inline Tensor1D tensor1d_subview(const Tensor1D& t, size_t elem_offset, size_t elem_count) {
    Tensor1D out;
    out.dtype = t.dtype;
    out.bytes_per_elem = t.bytes_per_elem;
    out.scale = t.scale;
    out.offset = t.offset;
    if (!t.valid() || elem_offset >= t.count) {
        out.data_ptr = nullptr;
        out.count = 0;
        out.bytes = 0;
        return out;
    }
    elem_count = std::min(elem_count, t.count - elem_offset);
    out.count = elem_count;
    out.bytes = out.bytes_per_elem * out.count;
    out.data_ptr = (void*)((uint8_t*)t.data_ptr + elem_offset * t.bytes_per_elem);
    return out;
}

inline float tensor1d_get_f32(const Tensor1D& t, size_t idx) {
    if (!t.data_ptr || idx >= t.count) return 0.0f;
    switch (t.dtype) {
        case TensorDType::F32:
            return ((const float*)t.data_ptr)[idx];
        case TensorDType::F16:
            return (float)((const half_float::half*)t.data_ptr)[idx];
        default:
            return 0.0f;
    }
}

inline void tensor1d_set_f32(Tensor1D& t, size_t idx, float v) {
    if (!t.data_ptr || idx >= t.count) return;
    switch (t.dtype) {
        case TensorDType::F32:
            ((float*)t.data_ptr)[idx] = v;
            return;
        case TensorDType::F16:
            ((half_float::half*)t.data_ptr)[idx] = (half_float::half)v;
            return;
        default:
            return;
    }
}

inline void tensor1d_add_bias(Tensor1D& t, size_t idx, float bias) {
    const float cur = tensor1d_get_f32(t, idx);
    tensor1d_set_f32(t, idx, cur + bias);
}

} // namespace rwkvmobile

#endif // RWKVMOBILE_TENSOR_H


