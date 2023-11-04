#ifndef BLADE_MEMORY_SHAPE_HH
#define BLADE_MEMORY_SHAPE_HH

#include <array>

#include "blade/memory/types.hh"

namespace Blade {

template<U64 Dimensions>
struct Shape {
 public:
    using Type = std::array<U64, Dimensions>;

    __host__ __device__ Shape() : _shape({0}), _size(0) {};
    __host__ __device__ Shape(const Type& shape) : _shape(shape) { computeCache(); }
    __host__ __device__ Shape(const Shape& shape) : _shape(shape._shape) { computeCache(); }

    __host__ __device__ constexpr U64 size() const {
        return _size;
    }

    __host__ __device__ constexpr const U64* data() const {
        return _shape.data();
    }

    __host__ __device__ U64 shapeToOffset(Type shape, U64 axis = 0, U64 offset = 0) const {
        shape[axis] += offset;
        U64 index = 0;
        for (U64 i = 0; i < Dimensions; i++) {
            index += shape[i] * _strides[i];
        }
        return index;
    }

    __host__ __device__ Type offsetToShape(U64 index, U64 axis = 0, U64 offset = 0) const {
        Type shape;
        for (U64 i = 0; i < Dimensions; i++) {
            shape[i] = index / _strides[i];
            index -= shape[i] * _strides[i];
        }
        shape[axis] += offset;
        return shape;
    }

    __host__ __device__ constexpr U64 dimensions() const {
        return Dimensions;
    }

    __host__ __device__ bool operator==(const Shape& other) const {
        return _shape == other._shape;
    }

    __host__ __device__ bool operator!=(const Shape& other) const {
        return !(_shape == other._shape);
    }

    __host__ __device__ operator Type() {
        return _shape;
    }

    __host__ __device__ operator const Type() const {
        return _shape;
    }

    __host__ __device__ constexpr const U64& operator[](const U64& idx) const {
        return _shape[idx];
    }

 protected:
    Type _shape;
    Type _strides;
    U64 _size;

    __host__ __device__ Type operator*(const Shape& other) const {
        Type result = _shape;
        for (U64 i = 0; i < Dimensions; i++) {
            result[i] *= other._shape[i];
        }
        return result;
    }

    __host__ __device__ Type operator/(const Shape& other) const {
        Type result = _shape;
        for (U64 i = 0; i < Dimensions; i++) {
            result[i] /= other._shape[i];
        }
        return result;
    }

    __host__ __device__ void computeCache() {
        _size = 1;
        for (U64 i = 0; i < Dimensions; i++) {
            _size *= _shape[i];
        }

        for (U64 i = 0; i < Dimensions; i++) {
            _strides[i] = 1;
            for (U64 j = i + 1; j < Dimensions; j++) {
                _strides[i] *= _shape[j];
            }
        }
    }
};

}  // namespace Blade

#endif
