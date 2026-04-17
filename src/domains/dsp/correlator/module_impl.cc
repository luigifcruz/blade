#include "module_impl.hh"

namespace Jetstream::Modules {

Result CorrelatorImpl::validate() {
    const auto& config = *candidate();

    if (config.integrationRate == 0) {
        JST_ERROR("[MODULE_CORRELATOR] The integration rate must be positive.");
        return Result::ERROR;
    }

    if (config.conjugateAntennaIndex > 1) {
        JST_ERROR("[MODULE_CORRELATOR] Unsupported conjugate antenna index {}. Expected 0 or 1.",
                  config.conjugateAntennaIndex);
        return Result::ERROR;
    }

    if (config.calculationMode != "integer" &&
        config.calculationMode != "single_precision_fp" &&
        config.calculationMode != "double_precision_fp") {
        JST_ERROR("[MODULE_CORRELATOR] Unsupported calculation mode '{}'.",
                  config.calculationMode);
        return Result::ERROR;
    }

    if (config.blockSize == 0) {
        JST_ERROR("[MODULE_CORRELATOR] The CUDA block size must be positive.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result CorrelatorImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer"));
    JST_CHECK(defineInterfaceOutput("buffer"));

    return Result::SUCCESS;
}

Result CorrelatorImpl::create() {
    inputTensor = inputs().at("buffer").tensor;

    if (inputTensor.rank() != kExpectedRank) {
        JST_ERROR("[MODULE_CORRELATOR] Input tensor must have {} dimensions [A, F, T, P], but received shape {}.",
                  kExpectedRank,
                  inputTensor.shape());
        return Result::ERROR;
    }

    if (!inputTensor.contiguous()) {
        JST_ERROR("[MODULE_CORRELATOR] Input tensor must be contiguous.");
        return Result::ERROR;
    }

    if (inputTensor.shape(kAspectAxis) == 0) {
        JST_ERROR("[MODULE_CORRELATOR] Input aspect dimension must be positive.");
        return Result::ERROR;
    }

    if (inputTensor.shape(kFrequencyAxis) == 0) {
        JST_ERROR("[MODULE_CORRELATOR] Input frequency dimension must be positive.");
        return Result::ERROR;
    }

    if (inputTensor.shape(kPolarizationAxis) != kExpectedInputPolarizations) {
        JST_ERROR("[MODULE_CORRELATOR] Input polarization dimension must be {}, but received {}.",
                  kExpectedInputPolarizations,
                  inputTensor.shape(kPolarizationAxis));
        return Result::ERROR;
    }

    optimizeTimeDomain = inputTensor.shape(kTimeAxis) > inputTensor.shape(kFrequencyAxis);
    blockSizeX = optimizeTimeDomain ? 1 : blockSize;
    blockSizeY = optimizeTimeDomain ? blockSize : 1;

    if ((inputTensor.shape(kFrequencyAxis) % blockSizeX) != 0) {
        JST_ERROR("[MODULE_CORRELATOR] Input frequency dimension {} is not divisible by the block size {}.",
                  inputTensor.shape(kFrequencyAxis),
                  blockSizeX);
        return Result::ERROR;
    }

    if ((inputTensor.shape(kTimeAxis) % blockSizeY) != 0) {
        JST_ERROR("[MODULE_CORRELATOR] Input time dimension {} is not divisible by the block size {}.",
                  inputTensor.shape(kTimeAxis),
                  blockSizeY);
        return Result::ERROR;
    }

    sharedMemoryEnabled = useSharedMemory && optimizeTimeDomain;

    baselineCount = (inputTensor.shape(kAspectAxis) * (inputTensor.shape(kAspectAxis) + 1)) / 2;
    const Shape outputShape = {
        baselineCount,
        inputTensor.shape(kFrequencyAxis),
        1,
        kOutputPolarizations,
    };

    JST_CHECK(outputTensor.create(inputTensor.device(), DataType::CF32, outputShape));
    JST_CHECK(outputTensor.propagateAttributes(inputTensor));

    outputs()["buffer"].produced(name(), "buffer", outputTensor);

    integrationStep = 0;

    return Result::SUCCESS;
}

Result CorrelatorImpl::destroy() {
    inputTensor = {};
    outputTensor = {};
    baselineCount = 0;
    blockSizeX = 0;
    blockSizeY = 0;
    integrationStep = 0;
    optimizeTimeDomain = false;
    sharedMemoryEnabled = false;

    return Result::SUCCESS;
}

Result CorrelatorImpl::reconfigure() {
    return Result::RECREATE;
}

}  // namespace Jetstream::Modules
