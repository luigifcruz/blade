#ifndef BLADE_MODULES_BEAMFORMER_VLA_HH
#define BLADE_MODULES_BEAMFORMER_VLA_HH

#include "blade/modules/beamformer/generic.hh"

namespace Blade::Modules::Beamformer {

template<typename IT, typename OT>
class BLADE_API VLA : public Generic<IT, OT> {
 public:
    explicit VLA(const typename Generic<IT, OT>::Config& config,
                 const typename Generic<IT, OT>::Input& input);

    constexpr U64 getInputSize() const {
        return this->config.numberOfAntennas *
               this->config.numberOfFrequencyChannels * 
               this->config.numberOfTimeSamples *
               this->config.numberOfPolarizations;
    }

    constexpr U64 getOutputSize() const {
        return this->config.numberOfBeams *
               this->config.numberOfTimeSamples *
               this->config.numberOfFrequencyChannels *
               this->config.numberOfPolarizations;
    }

    constexpr U64 getPhasorsSize() const {
        return this->config.numberOfBeams *
               this->config.numberOfAntennas *
               this->config.numberOfFrequencyChannels *
               this->config.numberOfPolarizations;
    }
};

}  // namespace Blade::Modules::Beamformer

#endif
