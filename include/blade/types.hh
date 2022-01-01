#ifndef BLADE_TYPES_HH
#define BLADE_TYPES_HH

#include <cuda_runtime.h>

#include <span>
#include <complex>

namespace Blade {

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
