#ifndef BLADE_TELESCOPES_GENERIC_BEAMFORMER_H
#define BLADE_TELESCOPES_GENERIC_BEAMFORMER_H

#include "blade/telescopes/generic/base.hh"

namespace Blade::Telescope::Generic::Beamformer {

class BLADE_API Test : protected Utils {
public:
    virtual ~Test() = default;

    virtual Result beamform() = 0;
    virtual std::span<const std::complex<int8_t>> getInputData() = 0;
    virtual std::span<const std::complex<float>> getPhasorsData() = 0;
    virtual std::span<const std::complex<float>> getOutputData() = 0;
};

} // namespace Blade::Telescope::Generic::Beamformer

#endif
