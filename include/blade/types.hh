#ifndef BLADE_TYPES_HH
#define BLADE_TYPES_HH

#include <cuda_runtime.h>

#include <span>
#include <complex>

#include "blade_config.hh"

#include "blade/memory/base.hh"

namespace Blade {

class ArrayTensorDimensions : public Dimensions {
 public:
    using Dimensions::Dimensions;

    constexpr const U64& numberOfAspects() const {
        return (*this)[0];
    }

    constexpr const U64& numberOfFrequencyChannels() const {
        return (*this)[1];
    }

    constexpr const U64& numberOfTimeSamples() const {
        return (*this)[2];
    }

    constexpr const U64& numberOfPolarizations() const {
        return (*this)[3];
    }
};

template<Device I, typename T>
using ArrayTensor = Vector<I, T, ArrayTensorDimensions>;

class PhasorTensorDimensions : public Dimensions {
 public:
    using Dimensions::Dimensions;

    constexpr const U64& numberOfBeams() const {
        return (*this)[0];
    }

    constexpr const U64& numberOfAntennas() const {
        return (*this)[1];
    }

    constexpr const U64& numberOfFrequencyChannels() const {
        return (*this)[2];
    }

    constexpr const U64& numberOfTimeSamples() const {
        return (*this)[3];
    }

    constexpr const U64& numberOfPolarizations() const {
        return (*this)[4];
    }
};

template<Device I, typename T>
using PhasorTensor = Vector<I, T, PhasorTensorDimensions>;

class DelayTensorDimensions : public Dimensions {
 public:
    using Dimensions::Dimensions;

    constexpr const U64& numberOfBeams() const {
        return (*this)[0];
    }

    constexpr const U64& numberOfAntennas() const {
        return (*this)[1];
    }
};

template<Device I, typename T>
using DelayTensor = Vector<I, T, DelayTensorDimensions>;

struct XYZ {
    double X;
    double Y;
    double Z;
};

struct UVW {
    double U;
    double V;
    double W;
};

struct LLA { 
    double LON;
    double LAT;
    double ALT;
};

struct RA_DEC {
    double RA;
    double DEC;
};

struct HA_DEC {
    double HA;
    double DEC;
};

}  // namespace Blade

#endif
