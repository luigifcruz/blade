#ifndef BLADE_MACROS_HH
#define BLADE_MACROS_HH

#include <stdint.h>
#include <math.h>

#include "blade/memory/types.hh"

namespace Blade {

enum class Taint : uint8_t {
    NONE     = 0 << 0,
    // Consumes externally allocated input data.
    CONSUMER = 1 << 0,
    // Produces internally allocated output data.
    PRODUCER = 1 << 1,
    // Modifies externally allocated input data.
    MODIFIER = 1 << 2,
    // Has a compute ratio other than one.
    CHRONOUS = 1 << 3, 
};

inline constexpr Taint operator|(Taint lhs, Taint rhs) {
    return static_cast<Taint>(static_cast<uint8_t>(lhs) | static_cast<uint8_t>(rhs));
}

inline constexpr Taint operator&(Taint lhs, Taint rhs) {
    return static_cast<Taint>(static_cast<uint8_t>(lhs) & static_cast<uint8_t>(rhs));
}

enum class Result : uint8_t {
    SUCCESS = 0,
    ERROR = 1,
    CUDA_ERROR,
    ASSERTION_ERROR,
    RUNNER_QUEUE_FULL,
    RUNNER_QUEUE_EMPTY,
    RUNNER_QUEUE_NONE_AVAILABLE,
    PIPELINE_EXHAUSTED,
};

}  // namespace Blade 

#ifndef BL_CUDA_CHECK_KERNEL
#define BL_CUDA_CHECK_KERNEL(callback) { \
    cudaError_t val; \
    if ((val = cudaPeekAtLastError()) != cudaSuccess) { \
        const char* err = cudaGetErrorString(val); \
        return callback(); \
    } \
}
#endif

#ifndef BL_CUDA_CHECK_KERNEL_THROW
#define BL_CUDA_CHECK_KERNEL_THROW(callback) { \
    cudaError_t val; \
    if ((val = cudaPeekAtLastError()) != cudaSuccess) { \
        const char* err = cudaGetErrorString(val); \
        throw callback(); \
    } \
}
#endif

#ifndef BL_CUDA_CHECK
#define BL_CUDA_CHECK(x, callback) { \
    cudaError_t val = (x); \
    if (val != cudaSuccess) { \
        const char* err = cudaGetErrorString(val); \
        callback(); \
        return Result::CUDA_ERROR; \
    } \
}
#endif

#ifndef BL_CUDA_CHECK_THROW
#define BL_CUDA_CHECK_THROW(x, callback) { \
    cudaError_t val = (x); \
    if (val != cudaSuccess) { \
        const char* err = cudaGetErrorString(val); \
        callback(); \
        throw Result::CUDA_ERROR; \
    } \
}
#endif

#ifndef BL_CUFFT_CHECK
#define BL_CUFFT_CHECK(x, callback) { \
    cufftResult err = (x); \
    if (err != CUFFT_SUCCESS) { \
        callback(); \
        return Result::CUDA_ERROR; \
    } \
}
#endif

#ifndef BL_CUFFT_CHECK_THROW
#define BL_CUFFT_CHECK_THROW(x, callback) { \
    cufftResult err = (x); \
    if (err != CUFFT_SUCCESS) { \
        callback(); \
        throw Result::CUDA_ERROR; \
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

#ifndef BL_CHECK_THROW
#define BL_CHECK_THROW(x) { \
    Result val = (x); \
    if (val != Result::SUCCESS) { \
        printf("Function %s (%s@%d) throwed!\n", __func__, __FILE__, __LINE__); \
        throw val; \
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
