#define BL_LOG_DOMAIN "M::BEAMFORMER::MEERKAT"

#include "blade/modules/beamformer/meerkat.hh"

namespace Blade::Modules::Beamformer {

template<typename IT, typename OT>
MeerKAT<IT, OT>::MeerKAT(const typename Generic<IT, OT>::Config& config,
                         const typename Generic<IT, OT>::Input& input)
        : Generic<IT, OT>(config, input) {
    this->grid = dim3(
        this->getInputBuffer().numberOfFrequencyChannels(),
        this->getInputBuffer().numberOfTimeSamples()/ config.blockSize);

    this->kernel = 
        Template("MeerKAT")
            .instantiate(
                this->getInputPhasors().numberOfBeams(),
                this->getInputPhasors().numberOfAntennas(),
                this->getInputBuffer().numberOfFrequencyChannels(),
                this->getInputBuffer().numberOfTimeSamples(),
                this->getInputBuffer().numberOfPolarizations(),
                config.blockSize,
                config.enableIncoherentBeam,
                config.enableIncoherentBeamSqrt);

    BL_CHECK_THROW(this->output.buf.resize(getOutputDims()));
}

template class BLADE_API MeerKAT<CF32, CF32>;

}  // namespace Blade::Modules::Beamformer
