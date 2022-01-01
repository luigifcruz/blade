#ifndef BLADE_MEMORY_TYPES_HH
#define BLADE_MEMORY_TYPES_HH

#include <cuComplex.h>
#include <cuda_fp16.h>

#include <span>
#include <vector>

#include "blade/logger.hh"
#include "blade/types.hh"

namespace Blade {

enum class Device : uint8_t {
    CPU     = 1 << 0,
    CUDA    = 1 << 1,
    METAL   = 1 << 2,
    VULKAN  = 1 << 3,
};

inline constexpr const Device operator|(Device lhs, Device rhs) {
    return static_cast<Device>(static_cast<uint8_t>(lhs) | static_cast<uint8_t>(rhs));
}

typedef __half  F16;
typedef float   F32;
typedef int8_t  I8;
typedef int16_t I16;
typedef int32_t I32;

typedef std::complex<F16> CF16;
typedef std::complex<F32> CF32;
typedef std::complex<I8>  CI8;
typedef std::complex<I16> CI16;
typedef std::complex<I32> CI32;

}  // namespace Blade

#endif