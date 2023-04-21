#ifndef BLADE_MEMORY_OPS_HH
#define BLADE_MEMORY_OPS_HH

#include <sstream>

#include "blade/memory/types.hh"

// This is not meant to be a fully featured complex library.
// It's meant to be a replacement for cuComplex.h that supports
// half-precision operations. It ignores multiple std::complex
// standards for the sake of computational efficiency.

namespace Blade::ops {

template<typename T>
class alignas(2 * sizeof(T)) complex {
 public:
    using TwinType = typename TypeInfo<T>::twintype;
        
    __host__ __device__ complex() : _real(0), _imag(0) {}
    __host__ __device__ complex(T r) : _real(r), _imag(0) {}
    __host__ __device__ complex(T r, T i) : _real(r), _imag(i) {}
    __host__ __device__ complex(TwinType t) : _real(t.x), _imag(t.y) {}

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
        if constexpr (std::is_same<T, F16>::value) {
            return __heq(_real, rhs._real) && __heq(_imag, rhs._imag);
        } else {
            return _real == rhs._real && _imag == rhs._imag;
        }
    }

    __host__ __device__ bool operator!=(const complex<T>& rhs) const {
        if constexpr (std::is_same<T, F16>::value) {
            return __hne(_real, rhs._real) || __hne(_imag, rhs._imag);
        } else {
            return _real != rhs._real || _imag != rhs._imag;
        }
    }

    __host__ __device__ bool operator<(const complex<T>& rhs) const {
        if constexpr (std::is_same<T, F16>::value) {
            return __hlt(_real, rhs._real) || (__heq(_real, rhs._real) && __hlt(_imag, rhs._imag));
        } else {
            return (_real < rhs._real) || ((_real == rhs._real) && (_imag < rhs._imag));
        }
    }

    __host__ __device__ bool operator>(const complex<T>& rhs) const {
        if constexpr (std::is_same<T, F16>::value) {
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

    friend std::ostream& operator<<(std::ostream& os, const complex<T>& c) {
        std::stringstream ss;
        if constexpr (std::is_same<T, F16>::value) {
            ss << __half2float(c._real) << "+" << __half2float(c._imag) << "i";
        } else {
            ss << c._real << "+" << c._imag << "i";
        }
        return os << ss.str();
    }

 private:
    T _real;
    T _imag;
};
  
}  // namespace Blade::ops

#endif
