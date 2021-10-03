#ifndef BLADE_MACROS_H
#define BLADE_MACROS_H

#include "blade/types.hh"

#ifndef BL_CUDA_CHECK_KERNEL
#define BL_CUDA_CHECK_KERNEL() { \
    cudaError_t err; \
    if ((err = cudaPeekAtLastError()) != cudaSuccess) { \
        BL_FATAL("Kernel failed to execute: {}", cudaGetErrorString(err)); \
        return Result::CUDA_ERROR; \
    } \
}
#endif

#ifndef BL_CUDA_CHECK
#define BL_CUDA_CHECK(x, callback) { \
    cudaError_t val = (x); \
    if (val != cudaSuccess) { \
        callback(); \
        return Result::CUDA_ERROR; \
    } \
}
#endif

#ifndef BL_CHECK
#define BL_CHECK(x) { \
    Result val = (x); \
    if (val != Result::SUCCESS) { \
        return val; \
    } \
}
#endif

#ifndef BL_ASSERT
#define BL_ASSERT(x) { \
    bool val = (x); \
    if (val != true) { \
        return Result::ASSERTION_ERROR; \
    } \
}
#endif

#endif
