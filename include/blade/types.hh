#ifndef BLADE_TYPES_H
#define BLADE_TYPES_H

#include "blade/common.hh"

#define BLADE_API __attribute__((visibility("default")))

namespace Blade {

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

} // namespace Blade

#endif
