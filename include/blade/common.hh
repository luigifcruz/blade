#ifndef BLADE_COMMON_H
#define BLADE_COMMON_H

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cuda_fp16.h>
#include <complex>
#include <span>

namespace Blade {

typedef __half  F16;
typedef float   F32;
typedef int8_t  I8;
typedef int16_t I16;
typedef int32_t I32;

typedef std::complex<F16> CF16;
typedef std::complex<F32> CF32;
typedef std::complex<I8>  CI8;
typedef std::complex<I16> CI16;
typedef std::complex<I32> CI32;

enum class Result : uint8_t {
    SUCCESS = 0,
    ERROR = 1,
    CUDA_ERROR,
    PYTHON_ERROR,
    ASSERTION_ERROR,
};

template <typename E>
constexpr auto to_underlying(E e) noexcept {
    return static_cast<std::underlying_type_t<E>>(e);
}

struct ArrayDims {
    std::size_t NBEAMS;
    std::size_t NANTS;
    std::size_t NCHANS;
    std::size_t NTIME;
    std::size_t NPOLS;
};

enum class RegisterKind : unsigned int {
    Mapped = cudaHostRegisterMapped,
    ReadOnly = cudaHostRegisterReadOnly,
    Default = cudaHostRegisterDefault,
    Portable = cudaHostRegisterPortable,
};

enum class CopyKind : unsigned int {
    D2H = cudaMemcpyDeviceToHost,
    H2D = cudaMemcpyHostToDevice,
    D2D = cudaMemcpyDeviceToDevice,
    H2H = cudaMemcpyHostToHost,
};

}  // namespace Blade

#ifndef BLADE_API
#define BLADE_API __attribute__((visibility("default")))
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

#endif  // BLADE_INCLUDE_BLADE_COMMON_HH_
