#ifndef BLADE_MEMORY_OPS_HH
#define BLADE_MEMORY_OPS_HH

#include <cuda_fp16.h>

#if !defined(__CUDACC_RTC__) && !defined(BL_OPS_HOST_SIDE_KEY)
// This is not meant to be a fully featured complex library.
// It's meant to be a replacement for cuComplex.h that supports
// half-precision operations. It ignores multiple std::complex
// standards for the sake of computational efficiency.
#error "This header should only be included in device code."
#endif

namespace Blade::ops {

template<typename T>
class alignas(2 * sizeof(T)) complex {
 public:   
    __host__ __device__ complex() : _real(0), _imag(0) {}
    __host__ __device__ complex(T r) : _real(r), _imag(0) {}
    __host__ __device__ complex(T r, T i) : _real(r), _imag(i) {}

    __host__ __device__ complex<T> operator+(const complex<T>& rhs) const {
        return complex<T>(_real + rhs._real, _imag + rhs._imag);
    }

    __host__ __device__ complex<T> operator-(const complex<T>& rhs) const {
        return complex<T>(_real - rhs._real, _imag - rhs._imag);
    }

    __host__ __device__ complex<T> operator*(const complex<T>& rhs) const {
        return complex<T>(_real * rhs._real - _imag * rhs._imag, 
                          _real * rhs._imag + _imag * rhs._real);
    }

    __host__ __device__ complex<T> operator/(const complex<T>& rhs) const {
        T denom = rhs._real * rhs._real + rhs._imag * rhs._imag;
        T real = (_real * rhs._real + _imag * rhs._imag) / denom;
        T imag = (_imag * rhs._real - _real * rhs._imag) / denom;
        return complex<T>(real, imag);
    }

    __host__ __device__ bool operator==(const complex<T>& rhs) const {
        if constexpr (std::is_same<T, __half>::value) {
            return __heq(_real, rhs._real) && __heq(_imag, rhs._imag);
        } else {
            return _real == rhs._real && _imag == rhs._imag;
        }
    }

    __host__ __device__ bool operator!=(const complex<T>& rhs) const {
        if constexpr (std::is_same<T, __half>::value) {
            return __hne(_real, rhs._real) || __hne(_imag, rhs._imag);
        } else {
            return _real != rhs._real || _imag != rhs._imag;
        }
    }

    __host__ __device__ bool operator<(const complex<T>& rhs) const {
        if constexpr (std::is_same<T, __half>::value) {
            return __hlt(_real, rhs._real) || (__heq(_real, rhs._real) && __hlt(_imag, rhs._imag));
        } else {
            return (_real < rhs._real) || ((_real == rhs._real) && (_imag < rhs._imag));
        }
    }

    __host__ __device__ bool operator>(const complex<T>& rhs) const {
        if constexpr (std::is_same<T, __half>::value) {
            return __hgt(_real > rhs._real) || (__heq(_real, rhs._real) && __hgt(_imag, rhs._imag));
        } else {
            return (_real > rhs._real) || ((_real == rhs._real) && (_imag > rhs._imag));
        }
    }

    __host__ __device__ constexpr T real() const {
        return _real;
    }

    __host__ __device__ constexpr T imag() const {
        return _imag;
    }

 private:
    T _real;
    T _imag;
};
  
}  // namespace Blade::ops

namespace Blade::ops::types {

typedef __half   F16;
typedef float    F32;
typedef double   F64;
typedef int8_t   I8;
typedef int16_t  I16;
typedef int32_t  I32;
typedef int64_t  I64;
typedef uint8_t  U8;
typedef uint16_t U16;
typedef uint32_t U32;
typedef uint64_t U64;
typedef bool     BOOL;

typedef ops::complex<F16> CF16;
typedef ops::complex<F32> CF32;
typedef ops::complex<F64> CF64;
typedef ops::complex<I8>  CI8;
typedef ops::complex<I16> CI16;
typedef ops::complex<I32> CI32;
typedef ops::complex<I64> CI64;
typedef ops::complex<U8>  CU8;
typedef ops::complex<U16> CU16;
typedef ops::complex<U32> CU32;
typedef ops::complex<U64> CU64;

}  // namespace Blade::ops::types

#endif
