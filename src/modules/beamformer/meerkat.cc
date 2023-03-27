#define BL_LOG_DOMAIN "M::BEAMFORMER::MEERKAT"

#include "blade/modules/beamformer/meerkat.hh"

namespace Blade::Modules::Beamformer {

template<typename IT, typename OT>
MeerKAT<IT, OT>::MeerKAT(const typename Generic<IT, OT>::Config& config,
                         const typename Generic<IT, OT>::Input& input, 
                         const cudaStream_t& stream)
        : Generic<IT, OT>(config, input, stream) {
    // Configure kernels.
    BL_CHECK_THROW(
        this->createKernel(
            // Kernel name. 
            "main",
            // Kernel function key.
            "MeerKAT",
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

template class BLADE_API MeerKAT<CF32, CF32>;

}  // namespace Blade::Modules::Beamformer
