#define BL_LOG_DOMAIN "M::BEAMFORMER::ATA"

#include "blade/modules/beamformer/ata.hh"

namespace Blade::Modules::Beamformer {

template<typename IT, typename OT>
ATA<IT, OT>::ATA(const typename Generic<IT, OT>::Config& config,
                 const typename Generic<IT, OT>::Input& input)
        : Generic<IT, OT>(config, input) {
    if (config.numberOfBeams > config.blockSize) {
        BL_FATAL("The block size ({}) is smaller than the number "
                "of beams ({}).", config.blockSize, config.numberOfBeams);
        BL_CHECK_THROW(Result::ERROR);
    }

    this->grid = dim3(
        config.numberOfFrequencyChannels,
        config.numberOfTimeSamples / config.blockSize);

    this->kernel = 
        Template("ATA")
            .instantiate(
                config.numberOfBeams,
                config.numberOfAntennas,
                config.numberOfFrequencyChannels,
                config.numberOfTimeSamples,
                config.numberOfPolarizations,
                config.blockSize, 
                config.enableIncoherentBeam,
                config.enableIncoherentBeamSqrt);

    BL_CHECK_THROW(this->InitInput(this->input.buf, getInputSize()));
    BL_CHECK_THROW(this->InitInput(this->input.phasors, getPhasorsSize()));
    BL_CHECK_THROW(this->InitOutput(this->output.buf, getOutputSize()));
}

template class BLADE_API ATA<CF32, CF32>;

}  // namespace Blade::Modules::Beamformer
