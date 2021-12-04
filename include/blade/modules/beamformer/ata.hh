#ifndef BLADE_MODULES_BEAMFORMER_ATA_HH
#define BLADE_MODULES_BEAMFORMER_ATA_HH

#include "blade/modules/beamformer/generic.hh"

namespace Blade::Modules::Beamformer {

template<typename IT, typename OT>
class BLADE_API ATA : public Generic<IT, OT> {
 public:
    class Test;

    explicit ATA(const typename Generic<IT, OT>::Config& config,
                 const typename Generic<IT, OT>::Input& input);

    constexpr std::size_t getInputSize() const {
        return this->config.dims.NANTS * this->config.dims.NCHANS *
            this->config.dims.NTIME * this->config.dims.NPOLS;
    }

    constexpr std::size_t getOutputSize() const {
        return this->config.dims.NBEAMS * this->config.dims.NTIME *
            this->config.dims.NCHANS * this->config.dims.NPOLS;
    }

    constexpr std::size_t getPhasorsSize() const {
        return this->config.dims.NBEAMS * this->config.dims.NANTS *
            this->config.dims.NCHANS * this->config.dims.NPOLS;
    }
};

}  // namespace Blade::Modules::Beamformer

#endif
