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

struct ArrayDimensions {
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

    const bool operator==(const ArrayDimensions& other) const {
        return (other.A == this->A) &&
               (other.F == this->F) &&
               (other.T == this->T) &&
               (other.P == this->P);
    }

    ArrayDimensions operator*(const ArrayDimensions& other) const {
        return ArrayDimensions {
            other.A * this->A, 
            other.F * this->F,
            other.T * this->T,
            other.P * this->P,
        };
    }

    ArrayDimensions operator/(const ArrayDimensions& other) const {
        return ArrayDimensions {
            this->A / other.A, 
            this->F / other.F,
            this->T / other.T,
            this->P / other.P,
        };
    }
};

template<Device I, typename T>
using ArrayTensor = Vector<I, T, ArrayDimensions>;

struct PhasorDimensions {
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

    const bool operator==(const PhasorDimensions& other) const {
        return (other.B == this->B) &&
               (other.A == this->A) &&
               (other.F == this->F) &&
               (other.T == this->T) &&
               (other.P == this->P);
    }

    PhasorDimensions operator*(const PhasorDimensions& other) const {
        return PhasorDimensions {
            other.B * this->B, 
            other.A * this->A, 
            other.F * this->F,
            other.T * this->T,
            other.P * this->P,
        };
    }

    PhasorDimensions operator/(const PhasorDimensions& other) const {
        return PhasorDimensions {
            this->B / other.B, 
            this->A / other.A, 
            this->F / other.F,
            this->T / other.T,
            this->P / other.P,
        };
    }
};

template<Device I, typename T>
using PhasorTensor = Vector<I, T, PhasorDimensions>;

struct DelayDimensions {
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

    const bool operator==(const DelayDimensions& other) const {
        return (other.B == this->B) &&
               (other.A == this->A);
    }

    DelayDimensions operator*(const DelayDimensions& other) const {
        return DelayDimensions {
            other.B * this->B, 
            other.A * this->A, 
        };
    }

    DelayDimensions operator/(const DelayDimensions& other) const {
        return DelayDimensions {
            this->B / other.B, 
            this->A / other.A, 
        };
    }
};

template<Device I, typename T>
using DelayTensor = Vector<I, T, DelayDimensions>;

}  // namespace Blade

namespace fmt {

template <>
struct formatter<Blade::ArrayDimensions> {
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx) {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto format(const Blade::ArrayDimensions& p, FormatContext& ctx) {
        return format_to(ctx.out(), "[{}, {}, {}, {}]", p.A, p.F, p.T, p.P);
    }
};

template <>
struct formatter<Blade::PhasorDimensions> {
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx) {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto format(const Blade::PhasorDimensions& p, FormatContext& ctx) {
        return format_to(ctx.out(), "[{}, {}, {}, {}, {}]", p.B, p.A, p.F, p.T, p.P);
    }
};

template <>
struct formatter<Blade::DelayDimensions> {
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx) {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto format(const Blade::DelayDimensions& p, FormatContext& ctx) {
        return format_to(ctx.out(), "[{}, {}]", p.B, p.A);
    }
};

} // namespace fmt

#endif
