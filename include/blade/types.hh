#ifndef BLADE_TYPES_HH
#define BLADE_TYPES_HH

#include <cuda_runtime.h>

#include <span>
#include <complex>

#include "blade_config.hh"

namespace Blade {

enum class Result : uint8_t {
    SUCCESS = 0,
    ERROR = 1,
    CUDA_ERROR,
    ASSERTION_ERROR,
    EXHAUSTED,
    BUFFER_FULL,
    BUFFER_INCOMPLETE,
    BUFFER_EMPTY,
};

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
