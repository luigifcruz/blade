#include "blade/modules/beamformer/meerkat.hh"

namespace Blade::Modules::Beamformer {

template<typename IT, typename OT>
MeerKAT<IT, OT>::MeerKAT(const typename Generic<IT, OT>::Config& config,
                         const typename Generic<IT, OT>::Input& input)
        : Generic<IT, OT>(config, input) {
    this->grid = dim3(
        config.numberOfFrequencyChannels,
        config.numberOfTimeSamples / config.blockSize);

    this->kernel = 
        Template("MeerKAT")
            .instantiate(
                config.numberOfBeams,
                config.numberOfAntennas,
                config.numberOfFrequencyChannels,
                config.numberOfTimeSamples,
                config.numberOfPolarizations,
                config.blockSize);

    BL_CHECK_THROW(this->InitInput(this->input.buf, getInputSize()));
    BL_CHECK_THROW(this->InitInput(this->input.phasors, getPhasorsSize()));
    BL_CHECK_THROW(this->InitOutput(this->output.buf, getOutputSize()));
}

template class BLADE_API MeerKAT<CF32, CF32>;

}  // namespace Blade::Modules::Beamformer
