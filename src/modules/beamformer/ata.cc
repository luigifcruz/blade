#define BL_LOG_DOMAIN "M::BEAMFORMER::ATA"

#include "blade/modules/beamformer/ata.hh"

namespace Blade::Modules::Beamformer {

template<typename IT, typename OT>
ATA<IT, OT>::ATA(const typename Generic<IT, OT>::Config& config,
                 const typename Generic<IT, OT>::Input& input,
                 const cudaStream_t& stream)
        : Generic<IT, OT>(config, input, stream) {
    // Check configuration values.
    if (this->getInputPhasors().numberOfBeams() > config.blockSize) {
        BL_FATAL("The block size ({}) is smaller than the number of beams ({}).", 
                config.blockSize, this->getInputPhasors().numberOfBeams());
        BL_CHECK_THROW(Result::ERROR);
    }

    if (this->getInputPhasors().numberOfFrequencyChannels() != 
        this->getInputBuffer().numberOfFrequencyChannels()) {
        BL_FATAL("Number of frequency channels mismatch between phasors ({}) and buffer ({}).",
                this->getInputPhasors().numberOfFrequencyChannels(),
                this->getInputBuffer().numberOfFrequencyChannels());
        BL_CHECK_THROW(Result::ERROR);
    }

    if (this->getInputPhasors().numberOfPolarizations() != 
        this->getInputBuffer().numberOfPolarizations()) {
        BL_FATAL("Number of polarizations mismatch between phasors ({}) and buffer ({}).",
                this->getInputPhasors().numberOfPolarizations(),
                this->getInputBuffer().numberOfPolarizations());
        BL_CHECK_THROW(Result::ERROR);
    }

    if (this->getInputPhasors().numberOfAntennas() != 
        this->getInputBuffer().numberOfAspects()) {
        BL_FATAL("Number of antennas mismatch between phasors ({}) and buffer ({}).",
                this->getInputPhasors().numberOfAntennas(),
                this->getInputBuffer().numberOfAspects());
        BL_CHECK_THROW(Result::ERROR);
    }

    // Configure kernels.
    BL_CHECK_THROW(
        this->createKernel(
            // Kernel name. 
            "main",
            // Kernel function key.
            "ATA",
            // Kernel grid & block sizes.
            dim3(this->getInputBuffer().numberOfFrequencyChannels(),
                 this->getInputBuffer().numberOfTimeSamples() / config.blockSize),
            config.blockSize,
            // Kernel templates.
            this->getInputPhasors().numberOfBeams(),
            this->getInputPhasors().numberOfAntennas(),
            this->getInputBuffer().numberOfFrequencyChannels(),
            this->getInputBuffer().numberOfTimeSamples(),
            this->getInputBuffer().numberOfPolarizations(),
            config.blockSize,
            config.enableIncoherentBeam,
            config.enableIncoherentBeamSqrt
        )
    );

    // Allocate output buffers.
    this->output.buf = ArrayTensor<Device::CUDA, OT>(getOutputBufferShape());

    // Print configuration values.
    BL_INFO("Shape {} -> {}", this->getInputBuffer().str(), 
                              this->getOutputBuffer().str());
}

template class BLADE_API ATA<CF32, CF32>;

}  // namespace Blade::Modules::Beamformer
