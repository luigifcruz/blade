#ifndef BLADE_MODULES_PHASOR_ATA_HH
#define BLADE_MODULES_PHASOR_ATA_HH

#include "blade/modules/phasor/generic.hh"

namespace Blade::Modules::Phasor {

template<typename OT>
class BLADE_API ATA : public Generic<OT> {
 public:
    explicit ATA(const typename Generic<OT>::Config& config,
                 const typename Generic<OT>::Input& input);

    constexpr U64 getPhasorsSize() const {
        return this->config.numberOfAntennas *
               this->config.numberOfBeams * 
               this->config.numberOfFrequencyChannels *
               this->config.numberOfPolarizations;
    }
};

}  // namespace Blade::Modules::Phasor

#endif
