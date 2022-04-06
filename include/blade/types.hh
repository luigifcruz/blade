#ifndef BLADE_TYPES_HH
#define BLADE_TYPES_HH

#include <cuda_runtime.h>

#include <span>
#include <complex>

namespace Blade {

enum class Result : uint8_t {
    SUCCESS = 0,
    ERROR = 1,
    CUDA_ERROR,
    ASSERTION_ERROR,
};

struct XYZ {
    double X;
    double Y;
    double Z;
};

struct UVW {
    double U;
    double V;
    double Z;
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

struct ArrayDims {
    std::size_t NBEAMS;
    std::size_t NANTS;
    std::size_t NCHANS;
    std::size_t NTIME;
    std::size_t NPOLS;

    constexpr const std::size_t getNumberOfBeams() const {
        return this->NBEAMS;
    }

    constexpr const std::size_t getNumberOfAntennas() const {
        return this->NANTS;
    }

    constexpr const std::size_t getNumberOfChannels() const {
        return this->NCHANS;
    }

    constexpr const std::size_t getNumberOfTime() const {
        return this->NTIME;
    }

    constexpr const std::size_t getNumberOfPolarizations() const {
        return this->NPOLS;
    }

    constexpr const std::size_t getSize() const {
        return NBEAMS * NANTS * NCHANS * NTIME * NPOLS;
    }
};

}  // namespace Blade

#endif
