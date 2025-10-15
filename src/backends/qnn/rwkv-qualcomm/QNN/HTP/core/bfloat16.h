//==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef BFLOAT16_H
#define BFLOAT16_H

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cmath>
#include <limits>

#include "builtin_intrinsics.h"

#include "weak_linkage.h"
#include "macros_attribute.h"

PUSH_VISIBILITY(default)

struct API_EXPORT BFloat16 {
  public:
    constexpr BFloat16() : d(0) {}
    constexpr BFloat16(float f);
    constexpr BFloat16(const BFloat16 &f) : d(f.d) {}
    constexpr BFloat16 &operator=(const BFloat16 &f);
    constexpr BFloat16(BFloat16 &&f) = default;
    constexpr BFloat16 &operator=(BFloat16 &&f) = default;
    ~BFloat16() = default;

    constexpr bool is_zero() const;
    constexpr bool is_neg() const;
    constexpr bool is_inf() const;
    constexpr bool is_nan() const;
    constexpr bool is_subnorm() const;
    constexpr bool is_norm() const;
    constexpr bool is_finite() const;

    constexpr int16_t exp() const;
    constexpr uint16_t frac() const;
    constexpr uint16_t raw() const { return d; }

    static constexpr int exp_max() { return 127; }
    static constexpr int exp_min() { return -126; }
    static constexpr int16_t bias() { return 127; }

    static constexpr BFloat16 zero(bool neg = false);
    static constexpr BFloat16 qnan();
    static constexpr BFloat16 snan();
    static constexpr BFloat16 inf(bool neg = false);

    static constexpr BFloat16 from_raw(uint16_t v);

    constexpr operator float() const;

  private:
    union {
        uint16_t d;
        struct {
            uint16_t mantissa : 7;
            uint16_t exponent : 8;
            uint16_t sign : 1;
        };
    };

    constexpr uint16_t sign_bit() const;
    constexpr uint16_t exp_bits() const;
    constexpr uint16_t frac_bits() const;

    static constexpr uint16_t make(int sign, int exp, uint32_t frac);
    static constexpr uint32_t round(uint32_t v, unsigned s);

    friend API_FUNC_EXPORT BFloat16 operator-(BFloat16 a);
    friend API_FUNC_EXPORT BFloat16 operator+(BFloat16 a, BFloat16 b);
    friend API_FUNC_EXPORT BFloat16 operator-(BFloat16 a, BFloat16 b);
    friend API_FUNC_EXPORT BFloat16 operator*(BFloat16 a, BFloat16 b);
};

POP_VISIBILITY()

inline constexpr BFloat16 BFloat16::from_raw(uint16_t v)
{
    BFloat16 f;
    f.d = v;
    return f;
}

inline constexpr BFloat16::BFloat16(float f) : d(0)
{
    union U {
        constexpr U(float f) : as_f32(f) {}
        float as_f32;
        uint32_t as_u32;
    } const u(f);

    // Preserve NaN values
    // The only potential NaN values that can be lost are the ones that have an exp=0xFF and a non 0 bit in the 16 lsb
    if ((u.as_u32 & 0x7F80FFFF) > 0x7F800000) {
        d = 0x7FA0u; // qnan
        return;
    }

    // BFloat uses round to nearest even
    bool const neg = u.as_u32 & (uint32_t(1u) << 31u);
    int const exp_extract = (u.as_u32 >> 23u) & 0xFFu;
    uint32_t const frac_bits = u.as_u32 & 0x7FFFFFu;

    int const exp = exp_extract - 127;
    uint32_t frac = round(frac_bits | (uint32_t(1) << 23u), 23 - 7);
    d = make(neg, exp, frac);
}

inline constexpr BFloat16 &BFloat16::operator=(const BFloat16 &f)
{
    d = f.d;
    return *this;
}

inline constexpr uint16_t BFloat16::sign_bit() const
{
    return d & 0x8000u;
}

inline constexpr uint16_t BFloat16::exp_bits() const
{
    return d & 0x7F80u;
}

inline constexpr uint16_t BFloat16::frac_bits() const
{
    return d & 0x7Fu;
}

inline constexpr bool BFloat16::is_zero() const
{
    return (exp_bits() | frac_bits()) == 0x0000;
}

inline constexpr bool BFloat16::is_neg() const
{
    return sign_bit();
}

inline constexpr BFloat16 BFloat16::zero(bool neg)
{
    return BFloat16::from_raw((neg) ? 0x8000u : 0x0);
}

inline constexpr BFloat16 BFloat16::qnan()
{
    return BFloat16::from_raw(0x7FA0u);
}

inline constexpr BFloat16 BFloat16::snan()
{
    return BFloat16::from_raw(0x7FC0u); // impl defined
}

inline constexpr BFloat16 BFloat16::inf(bool neg)
{
    return BFloat16::from_raw((neg) ? 0xFF80u : 0x7F80u);
}

inline constexpr BFloat16::operator float() const
{
    union U {
        constexpr U(uint32_t u) : as_u32(u) {}
        float as_f32;
        uint32_t as_u32;
    } u(static_cast<uint32_t>(raw()) << 16);
    return u.as_f32;
}

inline constexpr bool BFloat16::is_norm() const
{
    return is_zero() || (!is_inf() && !is_nan() && !is_subnorm());
}

inline constexpr bool BFloat16::is_inf() const
{
    return exp_bits() == 0x7F80u && frac_bits() == 0x0u;
}

inline constexpr bool BFloat16::is_nan() const
{
    return exp_bits() == 0x7F80u && frac_bits() != 0x0u;
}

inline constexpr bool BFloat16::is_subnorm() const
{
    return exp_bits() == 0x0000 && frac_bits() != 0x0000;
}

inline constexpr bool BFloat16::is_finite() const
{
    return is_norm() || is_subnorm();
}

inline constexpr uint16_t BFloat16::make(int sign, int exp, uint32_t frac)
{
    assert(frac > 0);
#if defined(_MSC_VER)
    // HEX_COUNT_LEADING_ZERO as defined for MSVC is not a constexpr.
    // This logic is testing in gtest test_bfloat16.cc if changing this code please update test.
    unsigned clz = 32u;
    for (unsigned i = 0; i < 32; i++) {
        if (frac & (1u << (31u - i))) {
            clz = i;
            break;
        }
    }
#else
    unsigned const clz = static_cast<unsigned>(HEX_COUNT_LEADING_ZERO(frac));
#endif // _MSC_VER
    // For a finite, normalized non-zero number, clz should be 16+(16-8) = 24.
    int exp_inc = 24u - clz;
    if (exp + exp_inc > exp_max()) {
        // Number has a magnitude that is too large.
        return BFloat16::inf(sign).raw();
    }
    if (exp + exp_inc < exp_min()) {
        // This number can become subnormal or zero.
        // safe_rshift will hit an assert if the shift is out of range
        // If we had an out of range shift, then we should just clip it to the range
        // Which should cause the frac to become 0 in either case
        int mask = static_cast<int>(hnnx::get_safe_shift_mask<int>());
        int shift_amount = exp_min() - exp - exp_inc - 1;
        shift_amount = (shift_amount > mask) ? mask : shift_amount;
        frac = hnnx::safe_rshift(static_cast<unsigned>(frac), shift_amount);
        return (static_cast<uint16_t>(sign) << 15u) | (static_cast<uint16_t>(frac) & 0x007Fu); // 0 exp bits
    }

    if (exp_inc > 0) { // exp_inc < 0 not expected for float32 to bfloat16 casting
        frac = round(frac, static_cast<unsigned>(exp_inc));
        // Rounding can change the most significant bit, so check it again.
        // unsigned const clzr = HEX_COUNT_LEADING_ZERO(frac);
        // assert(clzr == 24);
        // clzr can only be 24 here because this make function is only called in the instantiation of a BFloat16 from a float.
        // In the current code path, there is a rounding of the fractional bits before it is passed into this make function.
        // As a result, there can be at most 8 significant bits in frac variable passed to the make function, which means that clzr can only be 24
        // However, if this function is called in other places where there are different limits on the number of significant bits in the input frac
        // then clzr may be 23 or 24 and that will need to be accounted for here.
    }
    exp += exp_inc;
    exp += bias();
    return (static_cast<uint16_t>(sign) << 15u) | (static_cast<uint16_t>(exp) << 7u) |
           (static_cast<uint16_t>(frac) & 0x007Fu);
}

inline constexpr uint32_t BFloat16::round(uint32_t v, unsigned s)
{
    assert(s > 0);
    unsigned const out_msb = hnnx::safe_lshift(1u, (s - 1));
    if ((v & out_msb) == 0) {
        // Round down.
        return hnnx::safe_rshift(v, s);
    }
    if ((v & (out_msb - 1)) == 0) {
        // It's a tie, round to even.
        v = hnnx::safe_rshift(v, s);
        return v & 1u ? v + 1 : v;
    }
    // Round up.
    return hnnx::safe_rshift(v, s) + 1;
}

inline constexpr uint16_t BFloat16::frac() const
{
    if (is_zero()) {
        return 0x0u;
    }
    uint16_t f = frac_bits();
    if (is_norm()) f |= 1u << 7u;
    return f;
}

inline constexpr int16_t BFloat16::exp() const
{
    int16_t const e = static_cast<int16_t>(exp_bits() >> 7u);
    return e != 0 ? e - bias() : exp_min();
}

PUSH_VISIBILITY(default)
template <> class API_EXPORT std::numeric_limits<BFloat16> {
  public:
    static constexpr bool is_specialized = true;
    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr bool has_infinity = true;
    static constexpr bool has_quiet_NaN = true;
    static constexpr bool has_signaling_NaN = true;
    static constexpr auto has_denorm = std::denorm_present;
    static constexpr bool has_denorm_loss = false; // libc++
    static constexpr auto round_style = std::round_to_nearest;
    static constexpr bool is_iec559 = false;
    static constexpr bool is_bounded = true;
    static constexpr bool is_modulo = false;
    static constexpr int digits = 8;
    static constexpr int digits10 = 2; // floor((digits-1) * log10(2))
    static constexpr int max_digits10 = 4; // ceil(digits * log10(2) + 1)
    static constexpr int radix = 2;
    static constexpr int min_exponent = -126;
    static constexpr int min_exponent10 = -37; // float32 value
    static constexpr int max_exponent = 127;
    static constexpr int max_exponent10 = 38; // largest finite val = 3.3895314E38
    static constexpr bool traps = false;
    static constexpr bool tinyness_before = false; // libc++

    static constexpr BFloat16 min() noexcept; // returns min positive normal
    static constexpr BFloat16 lowest() noexcept; // returns true min
    static constexpr BFloat16 max() noexcept; // max positive
    static constexpr BFloat16 epsilon() noexcept; // step at 1.0
    static constexpr BFloat16 round_error() noexcept; // 0.5
    static constexpr BFloat16 infinity() noexcept;
    static constexpr BFloat16 quiet_NaN() noexcept;
    static constexpr BFloat16 signaling_NaN() noexcept;
    static constexpr BFloat16 denorm_min() noexcept; // min positive denorm
};

POP_VISIBILITY()

constexpr BFloat16 std::numeric_limits<BFloat16>::min() noexcept
{
    // 0 0000 0001 0000000
    return BFloat16::from_raw(0x80u);
}

constexpr BFloat16 std::numeric_limits<BFloat16>::lowest() noexcept
{
    // -2^127 * (1.9921875)  ; 1 1111 1110 1111 111
    return BFloat16::from_raw(0xFF7Fu); // -3.3895314E38
}

constexpr BFloat16 std::numeric_limits<BFloat16>::max() noexcept
{
    return BFloat16::from_raw(0x7f7fu);
}

constexpr BFloat16 std::numeric_limits<BFloat16>::epsilon() noexcept
{
    // 2^-7 * (1)     ; 0 01111000 0000000
    return BFloat16::from_raw(0x3C00u); // next_after_1.0 - 1.0
}

constexpr BFloat16 std::numeric_limits<BFloat16>::round_error() noexcept
{
    // 2^-1 * (1)      ; 0 01111110 0000000
    return BFloat16::from_raw(0x3F00u); // 0.5
}

constexpr BFloat16 std::numeric_limits<BFloat16>::infinity() noexcept
{
    return BFloat16::inf(false);
}

constexpr BFloat16 std::numeric_limits<BFloat16>::quiet_NaN() noexcept
{
    return BFloat16::qnan();
}

constexpr BFloat16 std::numeric_limits<BFloat16>::signaling_NaN() noexcept
{
    return BFloat16::snan();
}

constexpr BFloat16 std::numeric_limits<BFloat16>::denorm_min() noexcept
{
    return BFloat16::from_raw(0x0001u);
}

#endif // BFLOAT16_H
