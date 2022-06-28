#ifndef BLADE_MODULES_BEAMFORMER_MEERKAT_HH
#define BLADE_MODULES_BEAMFORMER_MEERKAT_HH

#include "blade/modules/beamformer/generic.hh"

namespace Blade::Modules::Beamformer {

template<typename IT, typename OT>
class BLADE_API MeerKAT : public Generic<IT, OT> {
 public:
    explicit MeerKAT(const typename Generic<IT, OT>::Config& config,
                     const typename Generic<IT, OT>::Input& input);

    constexpr U64 getInputSize() const {
        return this->config.numberOfAntennas *
               this->config.numberOfFrequencyChannels * 
               this->config.numberOfTimeSamples *
               this->config.numberOfPolarizations;
    }

    constexpr U64 getOutputSize() const {
        return (
                    this->config.numberOfBeams +
                    (this->config.enableIncoherentBeam ? 1 : 0)
               ) *
               this->config.numberOfTimeSamples *
               this->config.numberOfFrequencyChannels *
               this->config.numberOfPolarizations;
    }

    constexpr U64 getPhasorsSize() const {
        return this->config.numberOfBeams *
               this->config.numberOfAntennas;
    }
};

}  // namespace Blade::Modules::Beamformer

#endif
