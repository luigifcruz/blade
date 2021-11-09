#ifndef BLADE_MODULES_BEAMFORMER_ATA_H
#define BLADE_MODULES_BEAMFORMER_ATA_H

#include "blade/modules/beamformer/generic.hh"

namespace Blade::Modules::Beamformer {

class BLADE_API ATA : public Generic {
 public:
    class Test;

    explicit ATA(const Config& config);

    constexpr std::size_t getInputSize() const {
        return config.dims.NANTS * config.dims.NCHANS *
            config.dims.NTIME * config.dims.NPOLS;
    }

    constexpr std::size_t getOutputSize() const {
        return config.dims.NBEAMS * config.dims.NTIME *
            config.dims.NCHANS * config.dims.NPOLS;
    }

    constexpr std::size_t getPhasorsSize() const {
        return config.dims.NBEAMS * config.dims.NANTS *
            config.dims.NCHANS * config.dims.NPOLS;
    }
};

}  // namespace Blade::Modules::Beamformer

#endif
