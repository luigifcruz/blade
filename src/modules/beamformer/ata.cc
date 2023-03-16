#define BL_LOG_DOMAIN "M::BEAMFORMER::ATA"

#include "blade/modules/beamformer/ata.hh"

namespace Blade::Modules::Beamformer {

template<typename IT, typename OT>
ATA<IT, OT>::ATA(const typename Generic<IT, OT>::Config& config,
                 const typename Generic<IT, OT>::Input& input,
                 const cudaStream_t& stream)
        : Generic<IT, OT>(config, input, stream) {
    // Check configuration values.
    if (this->getInputPhasors().shape().numberOfBeams() > config.blockSize) {
        BL_FATAL("The block size ({}) is smaller than the number of beams ({}).", 
                config.blockSize, this->getInputPhasors().shape().numberOfBeams());
        BL_CHECK_THROW(Result::ERROR);
    }

    if (this->getInputPhasors().shape().numberOfFrequencyChannels() != 
        this->getInputBuffer().shape().numberOfFrequencyChannels()) {
        BL_FATAL("Number of frequency channels mismatch between phasors ({}) and buffer ({}).",
                this->getInputPhasors().shape().numberOfFrequencyChannels(),
                this->getInputBuffer().shape().numberOfFrequencyChannels());
        BL_CHECK_THROW(Result::ERROR);
    }

    if (this->getInputPhasors().shape().numberOfPolarizations() != 
        this->getInputBuffer().shape().numberOfPolarizations()) {
        BL_FATAL("Number of polarizations mismatch between phasors ({}) and buffer ({}).",
                this->getInputPhasors().shape().numberOfPolarizations(),
                this->getInputBuffer().shape().numberOfPolarizations());
        BL_CHECK_THROW(Result::ERROR);
    }

    if (this->getInputPhasors().shape().numberOfAntennas() != 
        this->getInputBuffer().shape().numberOfAspects()) {
        BL_FATAL("Number of antennas mismatch between phasors ({}) and buffer ({}).",
                this->getInputPhasors().shape().numberOfAntennas(),
                this->getInputBuffer().shape().numberOfAspects());
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
            dim3(this->getInputBuffer().shape().numberOfFrequencyChannels(),
                 this->getInputBuffer().shape().numberOfTimeSamples() / config.blockSize),
            config.blockSize,
            // Kernel templates.
            this->getInputPhasors().shape().numberOfBeams(),
            this->getInputPhasors().shape().numberOfAntennas(),
            this->getInputBuffer().shape().numberOfFrequencyChannels(),
            this->getInputBuffer().shape().numberOfTimeSamples(),
            this->getInputBuffer().shape().numberOfPolarizations(),
            config.blockSize,
            config.enableIncoherentBeam,
            config.enableIncoherentBeamSqrt
        )
    );

    // Allocate output buffers.
    this->output.buf = ArrayTensor<Device::CUDA, OT>(getOutputBufferShape());

    // Print configuration values.
    BL_INFO("Shape: {} -> {}", this->getInputBuffer().shape(), 
                              this->getOutputBuffer().shape());
}

template class BLADE_API ATA<CF32, CF32>;

}  // namespace Blade::Modules::Beamformer
