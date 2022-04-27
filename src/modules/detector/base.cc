#include "blade/modules/detector.hh"

#include "detector.jit.hh"

namespace Blade::Modules {

template<typename IT, typename OT>
Detector<IT, OT>::Detector(const Config& config, const Input& input)
        : Module(config.blockSize, detector_kernel),
          config(config),
          input(input) {
    BL_INFO("===== Detector Module Configuration");

    if ((config.numberOfTimeSamples % config.integrationSize) != 0) {
        BL_FATAL("The number of time samples ({}) should be divisable "
                "by the integration size ({}).", config.numberOfTimeSamples,
                config.integrationSize);
        throw Result::ERROR;
    }

    if (config.numberOfOutputPolarizations != 4) {
        BL_FATAL("Number of output polarizations ({}) should be four (4).", config.numberOfOutputPolarizations);
        throw Result::ERROR;
    }

    if (config.numberOfPolarizations != 2) {
        BL_FATAL("Number of polarizations ({}) should be two (2).", config.numberOfPolarizations);
        throw Result::ERROR;
    }

    if (config.numberOfBeams <= 0) {
        BL_FATAL("Number of beams ({}) should be more than zero.", config.numberOfBeams);
        throw Result::ERROR;
    }

    if (config.integrationSize <= 0) {
        BL_WARN("Integration size ({}) should be more than zero.", config.integrationSize);
        throw Result::ERROR;
    }

    grid = dim3((((getInputSize() / config.numberOfPolarizations) + block.x - 1) / block.x));

    kernel =
        Template("detector")
            .instantiate(getInputSize() / 
                         config.numberOfPolarizations,
                         config.numberOfFrequencyChannels,
                         config.integrationSize); 

    BL_INFO("Number of Beams: {}", config.numberOfBeams);
    BL_INFO("Number of Frequency Channels: {}", config.numberOfFrequencyChannels);
    BL_INFO("Number of Time Samples: {}", config.numberOfTimeSamples);
    BL_INFO("Number of Polarizations: {}", config.numberOfPolarizations);
    BL_INFO("Integration Size: {}", config.integrationSize);
    BL_INFO("Number of Output Polarizations: {}", config.numberOfOutputPolarizations);

    BL_CHECK_THROW(InitInput(input.buf, getInputSize()));
    BL_CHECK_THROW(InitOutput(output.buf, getOutputSize()));
}

template<typename IT, typename OT>
Result Detector<IT, OT>::process(const cudaStream_t& stream) {
    cache
        .get_kernel(kernel)
        ->configure(grid, block, 0, stream)
        ->launch(input.buf.data(), output.buf.data());

    BL_CUDA_CHECK_KERNEL([&]{
        BL_FATAL("Module failed to execute: {}", err);
        return Result::CUDA_ERROR;
    });

    return Result::SUCCESS;
}

template class BLADE_API Detector<CF32, F32>;

}  // namespace Blade::Modules
