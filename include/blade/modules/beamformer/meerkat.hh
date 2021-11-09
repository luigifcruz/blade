#ifndef BLADE_MODULES_BEAMFORMER_MEERKAT_HH
#define BLADE_MODULES_BEAMFORMER_MEERKAT_HH

#include "blade/modules/beamformer/generic.hh"

namespace Blade::Modules::Beamformer {

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

}  // namespace Blade::Modules::Beamformer

#endif
