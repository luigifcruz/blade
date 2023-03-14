#ifndef BLADE_MEMORY_CUSTOM_HH
#define BLADE_MEMORY_CUSTOM_HH

#include <array>

#include "blade/memory/vector.hh"
#include "blade/memory/shape.hh"

namespace Blade {

struct ArrayShape : public Shape<4> {
 public:
    using Shape::Shape;

    constexpr const U64& numberOfAspects() const {
        return this->shape()[0];
    }

    constexpr const U64& numberOfFrequencyChannels() const {
        return this->shape()[1];
    }

    constexpr const U64& numberOfTimeSamples() const {
        return this->shape()[2];
    }

    constexpr const U64& numberOfPolarizations() const {
        return this->shape()[3];
    }

    ArrayShape operator*(const ArrayShape& other) const {
        return ArrayShape(Shape::operator*(other).shape());
    }

    ArrayShape operator/(const ArrayShape& other) const {
        return ArrayShape(Shape::operator/(other).shape());
    }

    const std::string str() const {
        return fmt::format("[A: {}, F: {}, T: {}, P: {}]", 
                           numberOfAspects(),
                           numberOfFrequencyChannels(),
                           numberOfTimeSamples(),
                           numberOfPolarizations()); 
    }
};

template<Device DeviceId, typename DataType>
using ArrayTensor = Vector<DeviceId, DataType, ArrayShape>;

struct PhasorShape : public Shape<5> {
 public:
    using Shape::Shape;

    constexpr const U64& numberOfBeams() const {
        return this->shape()[0];
    }

    constexpr const U64& numberOfAntennas() const {
        return this->shape()[1];
    }

    constexpr const U64& numberOfFrequencyChannels() const {
        return this->shape()[2];
    }

    constexpr const U64& numberOfTimeSamples() const {
        return this->shape()[3];
    }

    constexpr const U64& numberOfPolarizations() const {
        return this->shape()[4];
    }

    PhasorShape operator*(const PhasorShape& other) const {
        return PhasorShape(Shape::operator*(other).shape());
    }

    PhasorShape operator/(const PhasorShape& other) const {
        return PhasorShape(Shape::operator/(other).shape());
    }

    const std::string str() const {
        return fmt::format("[B: {}, A: {}, F: {}, T: {}, P: {}]", 
                           numberOfBeams(), 
                           numberOfAntennas(),
                           numberOfFrequencyChannels(),
                           numberOfTimeSamples(),
                           numberOfPolarizations()); 
    }
};

template<Device DeviceId, typename DataType>
using PhasorTensor = Vector<DeviceId, DataType, PhasorShape>;

struct DelayShape : public Shape<5> {
 public:
    using Shape::Shape;

    constexpr const U64& numberOfBeams() const {
        return this->shape()[0];
    }

    constexpr const U64& numberOfAntennas() const {
        return this->shape()[1];
    }

    DelayShape operator*(const DelayShape& other) const {
        return DelayShape(Shape::operator*(other).shape());
    }

    DelayShape operator/(const DelayShape& other) const {
        return DelayShape(Shape::operator/(other).shape());
    }

    const std::string str() const {
        return fmt::format("[B: {}, A: {}]", 
                           numberOfBeams(), 
                           numberOfAntennas()); 
    }
};

template<Device DeviceId, typename DataType>
using DelayTensor = Vector<DeviceId, DataType, DelayShape>;

template<Device DeviceId, typename DataType>
using Tensor = Vector<DeviceId, DataType, Shape<1>>;

}  // namespace Blade

#endif
