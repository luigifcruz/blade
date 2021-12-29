#ifndef BLADE_TYPES_HH
#define BLADE_TYPES_HH

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cuda_fp16.h>

#include <span>
#include <complex>

namespace Blade {

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

enum class Result : uint8_t {
    SUCCESS = 0,
    ERROR = 1,
    CUDA_ERROR,
    ASSERTION_ERROR,
};

struct ArrayDims {
    std::size_t NBEAMS;
    std::size_t NANTS;
    std::size_t NCHANS;
    std::size_t NTIME;
    std::size_t NPOLS;

    constexpr const std::size_t getSize() const {
        return NBEAMS * NANTS * NCHANS * NTIME * NPOLS;
    }
};

}  // namespace Blade

#endif
