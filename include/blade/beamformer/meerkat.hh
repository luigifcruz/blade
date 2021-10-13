#ifndef BLADE_BEAMFORMER_MEERKAT_H
#define BLADE_BEAMFORMER_MEERKAT_H

#include "blade/beamformer/generic.hh"

namespace Blade::Beamformer {

class BLADE_API MeerKAT : public Generic {
public:
    MeerKAT(const Config & config);

    constexpr std::size_t getInputSize() const {
        return config.NANTS*config.NCHANS*config.NTIME*config.NPOLS;
    };

    constexpr std::size_t getOutputSize() const {
        return config.NBEAMS*config.NTIME*config.NCHANS*config.NPOLS;
    };

    constexpr std::size_t getPhasorsSize() const {
        return config.NBEAMS*config.NANTS;
    };
};

} // namespace Blade::Beamformer

#endif
