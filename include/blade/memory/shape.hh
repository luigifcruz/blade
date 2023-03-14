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

    __host__ __device__ const U64 size() const {
        U64 size = 1;
        for (const auto& n : shape()) {
            size *= n;
        }
        return size; 
    }

    __host__ __device__ const U64 shapeToOffset(const Type& index) const {
        U64 offset = 0;

        for (U64 i = 0; i < index.size(); i++) {
            U64 product = index[i];

            for (U64 j = i + 1; j < index.size(); j++) {
                product *= shape()[j];
            }
            
            offset += product;
        }

        return offset;
    }

    __host__ __device__ const bool operator==(const Shape& other) const {
        bool result = false;
        for (U64 i = 0; i < shape().size(); i++) {
            result &= (shape()[i] == other.shape()[i]);
        }
        return result;
    }

    Shape operator*(const Shape& other) const {
        Type result = shape();
        for (U64 i = 0; i < shape().size(); i++) {
            result[i] *= other.shape()[i];
        }
        return result;
    }

    Shape operator/(const Shape& other) const {
        Type result = shape();
        for (U64 i = 0; i < shape().size(); i++) {
            result[i] /= other.shape()[i];
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
