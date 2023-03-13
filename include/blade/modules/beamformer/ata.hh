#ifndef BLADE_MODULES_BEAMFORMER_ATA_HH
#define BLADE_MODULES_BEAMFORMER_ATA_HH

#include "blade/modules/beamformer/generic.hh"

namespace Blade::Modules::Beamformer {

template<typename IT, typename OT>
class BLADE_API ATA : public Generic<IT, OT> {
 public:
    explicit ATA(const typename Generic<IT, OT>::Config& config,
                 const typename Generic<IT, OT>::Input& input,
                 const cudaStream_t& stream);

 protected:
    const ArrayShape getOutputBufferShape() const {
        return ArrayShape({
            this->getInputPhasors().numberOfBeams() 
                    + U64(this->config.enableIncoherentBeam ? 1 : 0),
            this->getInputBuffer().numberOfFrequencyChannels(),
            this->getInputBuffer().numberOfTimeSamples(),
            this->getInputBuffer().numberOfPolarizations(),
        });
    }
};

}  // namespace Blade::Modules::Beamformer

#endif
