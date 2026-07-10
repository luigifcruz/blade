#include "module_impl.hh"

namespace Jetstream::Modules {

Result DetectorImpl::validate() {
    const auto& config = *candidate();

    if (config.integrationRate == 0) {
        JST_ERROR("[MODULE_DETECTOR] The integration rate must be positive.");
        return Result::ERROR;
    }

    if (config.numberOfOutputPolarizations != 1 && config.numberOfOutputPolarizations != 4) {
        JST_ERROR("[MODULE_DETECTOR] Unsupported number of output polarizations {}. Expected 1 or 4.",
                  config.numberOfOutputPolarizations);
        return Result::ERROR;
    }

    if (config.blockSize == 0 || config.blockSize > 1024) {
        JST_ERROR("[MODULE_DETECTOR] The CUDA block size must be between 1 and 1024.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result DetectorImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer"));
    JST_CHECK(defineInterfaceOutput("buffer"));

    return Result::SUCCESS;
}

Result DetectorImpl::create() {
    inputTensor = inputs().at("buffer").tensor;

    if (inputTensor.rank() != kExpectedRank) {
        JST_ERROR("[MODULE_DETECTOR] Input tensor must have {} dimensions [A, F, T, P], but received shape {}.",
                  kExpectedRank,
                  inputTensor.shape());
        return Result::ERROR;
    }

    if (!inputTensor.contiguous()) {
        JST_ERROR("[MODULE_DETECTOR] Input tensor must be contiguous.");
        return Result::ERROR;
    }

    if (inputTensor.shape(kTimeAxis) % integrationRate != 0) {
        JST_ERROR("[MODULE_DETECTOR] Input time dimension {} is not divisible by the integration rate {}.",
                  inputTensor.shape(kTimeAxis),
                  integrationRate);
        return Result::ERROR;
    }

    if (inputTensor.shape(kPolarizationAxis) != kExpectedInputPolarizations) {
        JST_ERROR("[MODULE_DETECTOR] Input polarization dimension must be {}, but received {}.",
                  kExpectedInputPolarizations,
                  inputTensor.shape(kPolarizationAxis));
        return Result::ERROR;
    }

    if (inputTensor.shape(kAspectAxis) == 0) {
        JST_ERROR("[MODULE_DETECTOR] Input aspect dimension must be positive.");
        return Result::ERROR;
    }

    Shape outputShape = inputTensor.shape();
    outputShape[kTimeAxis] /= integrationRate;
    outputShape[kPolarizationAxis] = numberOfOutputPolarizations;

    JST_CHECK(outputTensor.create(inputTensor.device(), DataType::F32, outputShape));
    JST_CHECK(outputTensor.propagateAttributes(inputTensor));

    outputs()["buffer"].produced(name(), "buffer", outputTensor);

    inputSampleCount = inputTensor.size() / inputTensor.shape(kPolarizationAxis);

    return Result::SUCCESS;
}

Result DetectorImpl::destroy() {
    inputTensor = {};
    outputTensor = {};
    inputSampleCount = 0;

    return Result::SUCCESS;
}

Result DetectorImpl::reconfigure() {
    return Result::RECREATE;
}

}  // namespace Jetstream::Modules
