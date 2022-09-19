#define BL_LOG_DOMAIN "M::BEAMFORMER::ATA"

#include "blade/modules/beamformer/ata.hh"

namespace Blade::Modules::Beamformer {

template<typename IT, typename OT>
ATA<IT, OT>::ATA(const typename Generic<IT, OT>::Config& config,
                 const typename Generic<IT, OT>::Input& input)
        : Generic<IT, OT>(config, input) {
    // Check configuration values.
    if (this->getInputPhasors().dims().numberOfBeams() > config.blockSize) {
        BL_FATAL("The block size ({}) is smaller than the number of beams ({}).", 
                config.blockSize, this->getInputPhasors().dims().numberOfBeams());
        BL_CHECK_THROW(Result::ERROR);
    }

    if (this->getInputPhasors().dims().numberOfFrequencyChannels() != 
        this->getInputBuffer().dims().numberOfFrequencyChannels()) {
        BL_FATAL("Number of frequency channels mismatch between phasors ({}) and buffer ({}).",
                this->getInputPhasors().dims().numberOfFrequencyChannels(),
                this->getInputBuffer().dims().numberOfFrequencyChannels());
        BL_CHECK_THROW(Result::ERROR);
    }

    if (this->getInputPhasors().dims().numberOfPolarizations() != 
        this->getInputBuffer().dims().numberOfPolarizations()) {
        BL_FATAL("Number of polarizations mismatch between phasors ({}) and buffer ({}).",
                this->getInputPhasors().dims().numberOfPolarizations(),
                this->getInputBuffer().dims().numberOfPolarizations());
        BL_CHECK_THROW(Result::ERROR);
    }

    if (this->getInputPhasors().dims().numberOfAntennas() != 
        this->getInputBuffer().dims().numberOfAspects()) {
        BL_FATAL("Number of antennas mismatch between phasors ({}) and buffer ({}).",
                this->getInputPhasors().dims().numberOfAntennas(),
                this->getInputBuffer().dims().numberOfAspects());
        BL_CHECK_THROW(Result::ERROR);
    }

    // Configure kernels.
    this->grid = dim3(
        this->getInputBuffer().dims().numberOfFrequencyChannels(),
        this->getInputBuffer().dims().numberOfTimeSamples() / config.blockSize);

    this->kernel = 
        Template("ATA")
            .instantiate(
                this->getInputPhasors().dims().numberOfBeams(),
                this->getInputPhasors().dims().numberOfAntennas(),
                this->getInputBuffer().dims().numberOfFrequencyChannels(),
                this->getInputBuffer().dims().numberOfTimeSamples(),
                this->getInputBuffer().dims().numberOfPolarizations(),
                config.blockSize,
                config.enableIncoherentBeam,
                config.enableIncoherentBeamSqrt);

    // Allocate output buffers.
    BL_CHECK_THROW(this->output.buf.resize(getOutputBufferDims()));

    // Print configuration values.
    BL_INFO("Dimensions [A, F, T, P]: {} -> {}", this->getInputBuffer().dims(), 
                                                 this->getOutputBuffer().dims());
}

template class BLADE_API ATA<CF32, CF32>;

}  // namespace Blade::Modules::Beamformer
