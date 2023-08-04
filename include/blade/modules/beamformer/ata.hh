#ifndef BLADE_MODULES_BEAMFORMER_ATA_HH
#define BLADE_MODULES_BEAMFORMER_ATA_HH

#include "blade/modules/beamformer/generic.hh"

namespace Blade::Modules::Beamformer {

template<typename IT, typename OT>
class BLADE_API ATA : public Generic<IT, OT> {
 public:
    explicit ATA(const typename Generic<IT, OT>::Config& config,
                 const typename Generic<IT, OT>::Input& input,
                 const cudaStream_t& stream = 0);

 protected:
    const ArrayShape getOutputBufferShape() const {
        return ArrayShape({
            this->getInputPhasors().shape().numberOfBeams() 
                    + U64(this->config.enableIncoherentBeam ? 1 : 0),
            this->getInputBuffer().shape().numberOfFrequencyChannels(),
            this->getInputBuffer().shape().numberOfTimeSamples(),
            this->getInputBuffer().shape().numberOfPolarizations(),
        });
    }
};

}  // namespace Blade::Modules::Beamformer

#endif
