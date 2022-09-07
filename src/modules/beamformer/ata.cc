#define BL_LOG_DOMAIN "M::BEAMFORMER::ATA"

#include "blade/modules/beamformer/ata.hh"

namespace Blade::Modules::Beamformer {

template<typename IT, typename OT>
ATA<IT, OT>::ATA(const typename Generic<IT, OT>::Config& config,
                 const typename Generic<IT, OT>::Input& input)
        : Generic<IT, OT>(config, input) {
    if (this->getInputPhasors().numberOfBeams() > config.blockSize) {
        BL_FATAL("The block size ({}) is smaller than the number of beams ({}).", 
                config.blockSize, this->getInputPhasors().numberOfBeams());
        BL_CHECK_THROW(Result::ERROR);
    }

    this->grid = dim3(
        this->getInputBuffer().numberOfFrequencyChannels(),
        this->getInputBuffer().numberOfTimeSamples() / config.blockSize);

    this->kernel = 
        Template("ATA")
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

template class BLADE_API ATA<CF32, CF32>;

}  // namespace Blade::Modules::Beamformer
