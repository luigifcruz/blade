#ifndef BLADE_MEMORY_SHAPE_HH
#define BLADE_MEMORY_SHAPE_HH

#include <array>

#include "blade/memory/types.hh"

namespace Blade {

template<U64 Dimensions>
struct Shape {
 public:
    using Type = std::array<U64, Dimensions>;

    Shape() : _shape({0}) {};
    Shape(const Type& shape) : _shape(shape) {}
    Shape(const Shape& shape) : _shape(shape._shape) {}

    __host__ __device__ const U64 size() const {
        U64 size = 1;
        for (const auto& n : _shape) {
            size *= n;
        }
        return size; 
    }

    __host__ __device__ const U64 shapeToOffset(const Type& index) const {
        U64 offset = 0;

        for (U64 i = 0; i < index.size(); i++) {
            U64 product = index[i];

            for (U64 j = i + 1; j < index.size(); j++) {
                product *= _shape[j];
            }
            
            offset += product;
        }

        return offset;
    }

    constexpr const U64 dimensions() const {
        return Dimensions;
    }

    const bool operator!=(const Shape& other) const {
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

    Type operator*(const Shape& other) const {
        Type result = _shape;
        for (U64 i = 0; i < _shape.size(); i++) {
            result[i] *= other._shape[i];
        }
        return result;
    }

    Type operator/(const Shape& other) const {
        Type result = _shape;
        for (U64 i = 0; i < _shape.size(); i++) {
            result[i] /= other._shape[i];
        }
        return result;
    }
};

}  // namespace Blade

#endif
