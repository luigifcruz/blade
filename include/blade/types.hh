#ifndef BLADE_TYPES_H
#define BLADE_TYPES_H

#include <span>
#include <complex>

#include "blade/common.hh"

#define BLADE_API __attribute__((visibility("default")))

namespace Blade {

typedef std::span<std::complex<float>> VCF32;
typedef std::span<std::complex<int8_t>> VCI8;
typedef std::complex<float> CF32;
typedef std::complex<int8_t> CI8;
typedef std::span<float> VF32;
typedef std::span<int8_t> VI8;

enum class Result : uint8_t {
    SUCCESS = 0,
    ERROR = 1,
    CUDA_ERROR,
    PYTHON_ERROR,
    ASSERTION_ERROR,
};

struct ArrayDims {
 public:
    std::size_t NBEAMS;
    std::size_t NANTS;
    std::size_t NCHANS;
    std::size_t NTIME;
    std::size_t NPOLS;
};

}  // namespace Blade

#endif  // BLADE_INCLUDE_BLADE_TYPES_HH_
