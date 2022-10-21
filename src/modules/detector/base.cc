#define BL_LOG_DOMAIN "M::DETECTOR"

#include "blade/modules/detector.hh"

#include "detector.jit.hh"

namespace Blade::Modules {

template<typename IT, typename OT>
Detector<IT, OT>::Detector(const Config& config, const Input& input)
        : Module(detector_program),
          config(config),
          input(input) {
    // Check configuration values.
    if (config.integrationSize <= 0) {
        BL_WARN("Integration size ({}) should be more than zero.", config.integrationSize);
        BL_CHECK_THROW(Result::ERROR);
    }

    if ((getInputBuffer().dims().numberOfTimeSamples() % config.integrationSize) != 0) {
        BL_FATAL("The number of time samples ({}) should be divisable "
                "by the integration size ({}).", getInputBuffer().dims().numberOfTimeSamples(), config.integrationSize);
        BL_CHECK_THROW(Result::ERROR);
    }

    if (getInputBuffer().dims().numberOfPolarizations() != 2) {
        BL_FATAL("Number of polarizations ({}) should be two (2).", getInputBuffer().dims().numberOfPolarizations());
        BL_CHECK_THROW(Result::ERROR);
    }

    if (getInputBuffer().dims().numberOfAspects() <= 0) {
        BL_FATAL("Number of aspects ({}) should be more than zero.", getInputBuffer().dims().numberOfAspects());
        BL_CHECK_THROW(Result::ERROR);
    }

    // Configure kernel instantiation.
    std::string kernel_key;
    switch (config.numberOfOutputPolarizations) {
        case 4: kernel_key = "detector_4pol"; break;
        case 1: kernel_key = "detector_1pol"; break;
        default:
            BL_FATAL("Number of output polarizations ({}) not supported.", 
                config.numberOfOutputPolarizations);
            BL_CHECK_THROW(Result::ERROR);
    }

    BL_CHECK_THROW(
        createKernel(
            // Kernel name.
            "main",
            // Kernel function key.
            kernel_key,
            // Kernel grid & block size.
            PadGridSize(
                getInputBuffer().size() / 
                getInputBuffer().dims().numberOfPolarizations(), config.blockSize),
            config.blockSize,
            // Kernel templates.
            getInputBuffer().size() / getInputBuffer().dims().numberOfPolarizations(),
            config.integrationSize
        )
    );

    // Allocate output buffers.
    BL_CHECK_THROW(output.buf.resize(getOutputBufferDims()));

    // Print configuration values.
    BL_INFO("Type: {} -> {}", TypeInfo<IT>::name, TypeInfo<OT>::name);
    BL_INFO("Dimensions [A, F, T, P]: {} -> {}", getInputBuffer().dims(), getOutputBuffer().dims());
    BL_INFO("Integration Size: {}", config.integrationSize);
    BL_INFO("Number of Output Polarizations: {}", config.numberOfOutputPolarizations);
}

template<typename IT, typename OT>
const Result Detector<IT, OT>::process(const cudaStream_t& stream) {
    return runKernel("main", stream, input.buf.data(), output.buf.data());
}

template class BLADE_API Detector<CF32, F32>;

}  // namespace Blade::Modules
