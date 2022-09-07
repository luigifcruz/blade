#ifndef BLADE_MODULES_BEAMFORMER_MEERKAT_HH
#define BLADE_MODULES_BEAMFORMER_MEERKAT_HH

#include "blade/modules/beamformer/generic.hh"

namespace Blade::Modules::Beamformer {

template<typename IT, typename OT>
class BLADE_API MeerKAT : public Generic<IT, OT> {
 public:
    explicit MeerKAT(const typename Generic<IT, OT>::Config& config,
                     const typename Generic<IT, OT>::Input& input);

 protected:
    const ArrayTensorDimensions getOutputDims() const {
        return {
            this->getInputBuffer().numberOfAspects() 
                + U64(this->config.enableIncoherentBeam ? 1 : 0),
            this->getInputBuffer().numberOfFrequencyChannels(),
            this->getInputBuffer().numberOfTimeSamples(),
            this->getInputBuffer().numberOfPolarizations(),
        };
    }
};

}  // namespace Blade::Modules::Beamformer

#endif
