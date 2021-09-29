#ifndef BL_TYPE_H
#define BL_TYPE_H

#include <complex>
#include <iostream>
#include <cassert>
#include <cstddef>

#include <cuda_runtime.h>
#include <cuComplex.h>

namespace BL {

enum class Result : uint8_t {
    SUCCESS = 0,
    ERROR = 1,
};

} // namespace BL

#endif
