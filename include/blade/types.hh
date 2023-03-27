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

}  // namespace Blade

#endif
