#define BL_LOG_DOMAIN "M::BEAMFORMER::MEERKAT"

#include "blade/modules/beamformer/meerkat.hh"

namespace Blade::Modules::Beamformer {

template<typename IT, typename OT>
MeerKAT<IT, OT>::MeerKAT(const typename Generic<IT, OT>::Config& config,
                         const typename Generic<IT, OT>::Input& input)
        : Generic<IT, OT>(config, input) {
    // Configure kernels.
    BL_CHECK_THROW(
        this->createKernel(
            // Kernel name. 
            "main",
            // Kernel function key.
            "MeerKAT",
            // Kernel grid & block sizes.
            dim3(this->getInputBuffer().dims().numberOfFrequencyChannels(),
                 this->getInputBuffer().dims().numberOfTimeSamples() / config.blockSize),
            config.blockSize,
            // Kernel templates.
            this->getInputPhasors().dims().numberOfBeams(),
            this->getInputPhasors().dims().numberOfAntennas(),
            this->getInputBuffer().dims().numberOfFrequencyChannels(),
            this->getInputBuffer().dims().numberOfTimeSamples(),
            this->getInputBuffer().dims().numberOfPolarizations(),
            config.blockSize,
            config.enableIncoherentBeam,
            config.enableIncoherentBeamSqrt
        )
    );

    // Allocate output buffers.
    BL_CHECK_THROW(this->output.buf.resize(getOutputBufferDims()));

    // Print configuration values.
    BL_INFO("Dimensions [A, F, T, P]: {} -> {}", this->getInputBuffer().dims(), 
                                                 this->getOutputBuffer().dims());
}

template class BLADE_API MeerKAT<CF32, CF32>;

}  // namespace Blade::Modules::Beamformer
