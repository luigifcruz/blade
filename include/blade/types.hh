#ifndef BLADE_TYPES_H
#define BLADE_TYPES_H

#include <span>
#include <complex>
#include <type_traits>

#include "blade/common.hh"

#define BLADE_API __attribute__((visibility("default")))

namespace Blade {

typedef half F16;
typedef float F32;
typedef int8_t I8;
typedef int16_t I16;
typedef int32_t I32;

typedef std::complex<F16> CF16;
typedef std::complex<F32> CF32;
typedef std::complex<I8> CI8;
typedef std::complex<I16> CI16;
typedef std::complex<I32> CI32;

enum class Result : uint8_t {
    SUCCESS = 0,
    ERROR = 1,
    CUDA_ERROR,
    PYTHON_ERROR,
    ASSERTION_ERROR,
};

template <typename E>
constexpr auto to_underlying(E e) noexcept {
    return static_cast<std::underlying_type_t<E>>(e);
}

struct ArrayDims {
    std::size_t NBEAMS;
    std::size_t NANTS;
    std::size_t NCHANS;
    std::size_t NTIME;
    std::size_t NPOLS;
};

}  // namespace Blade

#endif  // BLADE_INCLUDE_BLADE_TYPES_HH_
