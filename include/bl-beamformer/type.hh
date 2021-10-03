#ifndef BL_TYPE_H
#define BL_TYPE_H

#include <complex>
#include <iostream>
#include <cassert>
#include <cstddef>
#include <span>

#include <cuda_runtime.h>
#include <cuComplex.h>

#define BL_API __attribute__((visibility("default")))

namespace BL {

enum class Result : uint8_t {
    SUCCESS = 0,
    ERROR = 1,
    CUDA_ERROR,
    ASSERTION_ERROR,
};

} // namespace BL

#endif
