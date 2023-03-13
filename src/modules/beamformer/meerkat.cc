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
    BL_INFO("Shape [A, F, T, P]: {} -> {}", this->getInputBuffer().shape(), 
                                            this->getOutputBuffer().shape());
}

template class BLADE_API MeerKAT<CF32, CF32>;

}  // namespace Blade::Modules::Beamformer
