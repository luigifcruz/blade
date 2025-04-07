#ifndef BLADE_TYPES_HH
#define BLADE_TYPES_HH

#include <cuda_runtime.h>

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

enum class POL : uint8_t {
    X,
    Y,
    L,
    R,
    XY,
    LR,
};

#ifndef __CUDA_ARCH__
inline std::ostream& operator<<(std::ostream& os, const POL& pol) {
    switch (pol) {
        case POL::X: return os << "X";
        case POL::Y: return os << "Y";
        case POL::L: return os << "L";
        case POL::R: return os << "R";
        case POL::XY: return os << "XY";
        case POL::LR: return os << "LR";
    }
    return os;
}
#endif

enum class CALC_MODE : uint8_t {
    INTEGER,
    SINGLE_PRECISION_FP,
    DOUBLE_PRECISION_FP
};

#ifndef __CUDA_ARCH__
inline std::ostream& operator<<(std::ostream& os, const CALC_MODE& mode) {
    switch (mode) {
        case CALC_MODE::INTEGER: return os << "Integer";
        case CALC_MODE::SINGLE_PRECISION_FP: return os << "Single Precision Floating Point";
        case CALC_MODE::DOUBLE_PRECISION_FP: return os << "Double Precision Floating Point";
    }
    return os;
}
#endif

}  // namespace Blade

#ifndef __CUDA_ARCH__
template <> struct bl::fmt::formatter<Blade::POL> : ostream_formatter {};
template <> struct bl::fmt::formatter<Blade::CALC_MODE> : ostream_formatter {};
#endif

#endif
