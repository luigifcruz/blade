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

} // namespace Blade

#endif
