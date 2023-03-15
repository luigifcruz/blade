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

    constexpr const Type& shape() const {
        return _shape;
    }

    const Result reshape(const Type& shape) {
        if (shape.size() != _shape.size()) {
            return Result::ERROR;
        }
        _shape = shape;

        return Result::SUCCESS;
    }

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

    __host__ __device__ const bool operator==(const Shape& other) const {
        return operator==(other.shape());
    }

    __host__ __device__ const bool operator==(const Type& other) const {
        bool result = true;
        for (U64 i = 0; i < _shape.size(); i++) {
            result &= (_shape[i] == other[i]);
        }
        BL_TRACE("{} {} {}", other, _shape, result);
        return result;
    }

    Shape operator*(const Shape& other) const {
        Type result = _shape;
        for (U64 i = 0; i < _shape.size(); i++) {
            result[i] *= other._shape[i];
        }
        return result;
    }

    Shape operator/(const Shape& other) const {
        Type result = _shape;
        for (U64 i = 0; i < _shape.size(); i++) {
            result[i] /= other._shape[i];
        }
        return result;
    }

    operator Type() const {
        return _shape;
    }

    const std::string str() const {
        return fmt::format("[{}]", _shape);
    };

 protected:
    Type _shape;
};

}  // namespace Blade


#endif
