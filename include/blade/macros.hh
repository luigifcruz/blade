#ifndef BLADE_MACROS_H
#define BLADE_MACROS_H

#include "blade/types.hh"

#ifndef BL_CUDA_CHECK_KERNEL
#define BL_CUDA_CHECK_KERNEL(callback) { \
    cudaError_t val; \
    if ((val = cudaPeekAtLastError()) != cudaSuccess) { \
        auto err = cudaGetErrorString(val); \
        return callback(); \
    } \
}
#endif

#ifndef BL_CUDA_CHECK
#define BL_CUDA_CHECK(x, callback) { \
    cudaError_t val = (x); \
    if (val != cudaSuccess) { \
        auto err = cudaGetErrorString(val); \
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

#ifndef BL_CATCH
#define BL_CATCH(x, callback) { \
    try { \
        (void)(x); \
    } catch (const std::exception& e) { \
        return callback(); \
    } \
}
#endif

#endif
