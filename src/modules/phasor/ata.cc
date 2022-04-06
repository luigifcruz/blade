#include "blade/modules/phasor/ata.hh"

namespace Blade::Modules::Phasor {

template<typename OT>
ATA<OT>::ATA(const typename Generic<OT>::Config& config,
             const typename Generic<OT>::Input& input)
        : Generic<OT>(config, input) {
    BL_CHECK_THROW(this->InitOutput(this->output.phasors, getPhasorsSize()));
}

template class BLADE_API ATA<CF32>;
template class BLADE_API ATA<CF64>;

}  // namespace Blade::Modules::Phasor
