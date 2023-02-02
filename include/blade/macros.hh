#ifndef BLADE_MACROS_HH
#define BLADE_MACROS_HH

#include <stdint.h>
#include <math.h>

namespace Blade {

enum class MemoryTaint : uint8_t {
    NONE     = 0 << 0,
    CONSUMER = 1 << 0,
    PRODUCER = 1 << 1,
    MODIFIER = 1 << 2,
};

inline constexpr MemoryTaint operator|(MemoryTaint lhs, MemoryTaint rhs) {
    return static_cast<MemoryTaint>(static_cast<uint8_t>(lhs) | static_cast<uint8_t>(rhs));
}

enum class Result : uint8_t {
    SUCCESS = 0,
    ERROR = 1,
    CUDA_ERROR,
    ASSERTION_ERROR,
    EXHAUSTED,
    BUFFER_FULL,
    BUFFER_INCOMPLETE,
    BUFFER_EMPTY,
    PLAN_SKIP_NO_SLOT,
    PLAN_SKIP_COMPUTE_INCOMPLETE,
    PLAN_SKIP_ACCUMULATION_INCOMPLETE,
    PLAN_SKIP_NO_DEQUEUE,
    PLAN_SKIP_USER_INITIATED,
    PLAN_ERROR_NO_SLOT,
    PLAN_ERROR_NO_ACCUMULATOR,
    PLAN_ERROR_ACCUMULATION_COMPLETE,
    PLAN_ERROR_DESTINATION_NOT_SYNCHRONIZED,
};

}  // namespace Blade 

#ifndef BL_PHYSICAL_CONSTANT_C
#define BL_PHYSICAL_CONSTANT_C (double)299792458.0 // Speed of Light (m/s)
#endif

#ifndef BL_PHYSICAL_CONSTANT_PI
#define BL_PHYSICAL_CONSTANT_PI M_PI
#endif

#ifndef BL_DEG_TO_RAD 
#define BL_DEG_TO_RAD(DEG) (DEG * M_PI / 180.0)
#endif

#ifndef BL_RAD_TO_DEG
#define BL_RAD_TO_DEG(RAD) (RAD * 180.0 / M_PI) 
#endif

#ifndef BLADE_API
#define BLADE_API __attribute__((visibility("default")))
#endif

#ifndef BLADE_HIDDEN
#define BLADE_HIDDEN __attribute__((visibility("hidden")))
#endif

#ifndef BL_CUDA_CHECK_KERNEL
#define BL_CUDA_CHECK_KERNEL(callback) { \
    cudaError_t val; \
    if ((val = cudaPeekAtLastError()) != cudaSuccess) { \
        auto err = cudaGetErrorString(val); \
        return callback(); \
    } \
}
#endif

#ifndef BL_CUDA_CHECK_KERNEL_THROW
#define BL_CUDA_CHECK_KERNEL_THROW(callback) { \
    cudaError_t val; \
    if ((val = cudaPeekAtLastError()) != cudaSuccess) { \
        auto err = cudaGetErrorString(val); \
        throw callback(); \
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

#ifndef BL_CUDA_CHECK_THROW
#define BL_CUDA_CHECK_THROW(x, callback) { \
    cudaError_t val = (x); \
    if (val != cudaSuccess) { \
        auto err = cudaGetErrorString(val); \
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
