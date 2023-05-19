#ifndef BLADE_MEMORY_SHAPE_HH
#define BLADE_MEMORY_SHAPE_HH

#include <array>

#include "blade/memory/types.hh"

namespace Blade {

template<U64 Dimensions>
struct Shape {
 public:
    using Type = std::array<U64, Dimensions>;

    Shape() : _shape({0}), _size(0) {};
    Shape(const Type& shape) : _shape(shape) { computeSize(); }
    Shape(const Shape& shape) : _shape(shape._shape) { computeSize(); }

    constexpr U64 size() const {
        return _size;
    }

    __host__ __device__ U64 shapeToOffset(const Type& index) const {
        U64 offset = 0;

        for (U64 i = 0; i < Dimensions; i++) {
            U64 product = index[i];

            for (U64 j = i + 1; j < Dimensions; j++) {
                product *= _shape[j];
            }
            
            offset += product;
        }

        return offset;
    }

    constexpr U64 dimensions() const {
        return Dimensions;
    }

    bool operator!=(const Shape& other) const {
        return !(_shape == other._shape);
    }

    operator Type() {
        return _shape;
    }

    operator const Type() const {
        return _shape;
    }

    constexpr const U64& operator[](U64 idx) const {
        return _shape[idx];
    }

 protected:
    Type _shape;
    U64 _size;

    Type operator*(const Shape& other) const {
        Type result = _shape;
        for (U64 i = 0; i < Dimensions; i++) {
            result[i] *= other._shape[i];
        }
        return result;
    }

    Type operator/(const Shape& other) const {
        Type result = _shape;
        for (U64 i = 0; i < Dimensions; i++) {
            result[i] /= other._shape[i];
        }
        return result;
    }

    void computeSize() {
        _size = 1;
        for (U64 i = 0; i < Dimensions; i++) {
            _size *= _shape[i];
        }
    }
};

}  // namespace Blade

#endif
