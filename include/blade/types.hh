#ifndef BLADE_TYPES_HH
#define BLADE_TYPES_HH

#include <cuda_runtime.h>

#include <span>
#include <complex>

#include "blade_config.hh"

#include "blade/memory/base.hh"

namespace Blade {

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

struct ArrayTensorDimensions {
 public:
    U64 A;
    U64 F;
    U64 T;
    U64 P;

    constexpr const U64& numberOfAspects() const {
        return A;
    }

    constexpr const U64& numberOfFrequencyChannels() const {
        return F;
    }

    constexpr const U64& numberOfTimeSamples() const {
        return T;
    }

    constexpr const U64& numberOfPolarizations() const {
        return P;
    }

    const U64 size() const {
        return A * F * T * P;
    }

    const bool operator==(const ArrayTensorDimensions& other) const {
        return (other.A == this->A) &&
               (other.F == this->F) &&
               (other.T == this->T) &&
               (other.P == this->P);
    }

    ArrayTensorDimensions operator*(const ArrayTensorDimensions& other) const {
        return ArrayTensorDimensions {
            other.A * this->A, 
            other.F * this->F,
            other.T * this->T,
            other.P * this->P,
        };
    }
};

template<Device I, typename T>
using ArrayTensor = Vector<I, T, ArrayTensorDimensions>;

struct ArrayCoefficientTensorDimensions {
 public:
    U64 A;
    U64 F;
    U64 P;

    constexpr const U64& numberOfAspects() const {
        return A;
    }

    constexpr const U64& numberOfFrequencyChannels() const {
        return F;
    }

    constexpr const U64& numberOfPolarizations() const {
        return P;
    }

    const U64 size() const {
        return A * F * P;
    }

    const bool operator==(const ArrayCoefficientTensorDimensions& other) const {
        return (other.A == this->A) &&
               (other.F == this->F) &&
               (other.P == this->P);
    }

    ArrayCoefficientTensorDimensions operator*(const ArrayCoefficientTensorDimensions& other) const {
        return ArrayCoefficientTensorDimensions {
            other.A * this->A, 
            other.F * this->F,
            other.P * this->P,
        };
    }
};

template<Device I, typename T>
using ArrayCoefficientTensor = Vector<I, T, ArrayCoefficientTensorDimensions>;

struct PhasorTensorDimensions {
 public:
    U64 B;
    U64 A;
    U64 F;
    U64 T;
    U64 P;

    constexpr const U64& numberOfBeams() const {
        return B;
    }

    constexpr const U64& numberOfAntennas() const {
        return A;
    }

    constexpr const U64& numberOfFrequencyChannels() const {
        return F;
    }

    constexpr const U64& numberOfTimeSamples() const {
        return T;
    }

    constexpr const U64& numberOfPolarizations() const {
        return P;
    }

    const U64 size() const {
        return B * A * F * T * P;
    }

    const bool operator==(const PhasorTensorDimensions& other) const {
        return (other.B == this->B) &&
               (other.A == this->A) &&
               (other.F == this->F) &&
               (other.T == this->T) &&
               (other.P == this->P);
    }

    PhasorTensorDimensions operator*(const PhasorTensorDimensions& other) const {
        return PhasorTensorDimensions {
            other.B * this->B, 
            other.A * this->A, 
            other.F * this->F,
            other.T * this->T,
            other.P * this->P,
        };
    }
};

template<Device I, typename T>
using PhasorTensor = Vector<I, T, PhasorTensorDimensions>;

struct DelayTensorDimensions {
 public:
    U64 B;
    U64 A;

    constexpr const U64& numberOfBeams() const {
        return B;
    }

    constexpr const U64& numberOfAntennas() const {
        return A;
    }

    const U64 size() const {
        return B * A;
    }

    const bool operator==(const DelayTensorDimensions& other) const {
        return (other.B == this->B) &&
               (other.A == this->A);
    }

    DelayTensorDimensions operator*(const DelayTensorDimensions& other) const {
        return DelayTensorDimensions {
            other.B * this->B, 
            other.A * this->A, 
        };
    }
};

template<Device I, typename T>
using DelayTensor = Vector<I, T, DelayTensorDimensions>;

}  // namespace Blade

namespace fmt {

template <>
struct formatter<Blade::ArrayTensorDimensions> {
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx) {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto format(const Blade::ArrayTensorDimensions& p, FormatContext& ctx) {
        return format_to(ctx.out(), "[{}, {}, {}, {}]", p.A, p.F, p.T, p.P);
    }
};

template <>
struct formatter<Blade::ArrayCoefficientTensorDimensions> {
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx) {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto format(const Blade::ArrayCoefficientTensorDimensions& p, FormatContext& ctx) {
        return format_to(ctx.out(), "[{}, {}, {}]", p.A, p.F, p.P);
    }
};

template <>
struct formatter<Blade::PhasorTensorDimensions> {
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx) {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto format(const Blade::PhasorTensorDimensions& p, FormatContext& ctx) {
        return format_to(ctx.out(), "[{}, {}, {}, {}, {}]", p.B, p.A, p.F, p.T, p.P);
    }
};

template <>
struct formatter<Blade::DelayTensorDimensions> {
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx) {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto format(const Blade::DelayTensorDimensions& p, FormatContext& ctx) {
        return format_to(ctx.out(), "[{}, {}]", p.B, p.A);
    }
};

} // namespace fmt

#endif
