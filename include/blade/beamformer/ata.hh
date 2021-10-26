#ifndef BLADE_BEAMFORMER_ATA_H
#define BLADE_BEAMFORMER_ATA_H

#include "blade/beamformer/generic.hh"

namespace Blade::Beamformer {

class BLADE_API ATA : public Generic {
public:
    class Test;

    explicit ATA(const Config& config);

    constexpr std::size_t getInputSize() const {
        return config.NANTS * config.NCHANS * config.NTIME * config.NPOLS;
    };

    constexpr std::size_t getOutputSize() const {
        return config.NBEAMS * config.NTIME * config.NCHANS * config.NPOLS;
    };

    constexpr std::size_t getPhasorsSize() const {
        return config.NBEAMS * config.NANTS * config.NCHANS * config.NPOLS;
    };
};

} // namespace Blade::Beamformer

#endif
