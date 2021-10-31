#ifndef BLADE_BEAMFORMER_MEERKAT_H
#define BLADE_BEAMFORMER_MEERKAT_H

#include "blade/beamformer/generic.hh"

namespace Blade::Beamformer {

class BLADE_API MeerKAT : public Generic {
 public:
    explicit MeerKAT(const Config& config);

    constexpr std::size_t getInputSize() const {
        return config.dims.NANTS * config.dims.NCHANS *
            config.dims.NTIME * config.dims.NPOLS;
    }

    constexpr std::size_t getOutputSize() const {
        return config.dims.NBEAMS * config.dims.NTIME *
            config.dims.NCHANS * config.dims.NPOLS;
    }

    constexpr std::size_t getPhasorsSize() const {
        return config.dims.NBEAMS * config.dims.NANTS;
    }
};

}  // namespace Blade::Beamformer

#endif  // BLADE_INCLUDE_BLADE_BEAMFORMER_MEERKAT_HH_
