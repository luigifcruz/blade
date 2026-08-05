#include "module_impl.hh"

#include <jetstream/tools/numeric.hh>

namespace Jetstream::Modules {

Result DetectorImpl::validate() {
    validatedSignalAxes = {};
    validatedOutputShape.clear();
    validatedSampleAxis = 0;
    validatedPolarizationAxis = 0;
    validatedInputSampleCount = 0;
    validatedOutputSizeBytes = 0;

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

    if (!inputs().contains("buffer")) {
        return Result::SUCCESS;
    }

    const Tensor& inputTensor = inputs().at("buffer").tensor;
    if (!inputTensor.validShape() || inputTensor.size() == 0) {
        return Result::SUCCESS;
    }

    if (inputTensor.rank() != kExpectedRank) {
        JST_ERROR("[MODULE_DETECTOR] Input tensor must have {} dimensions [A, F, T, P], but received shape {}.",
                  kExpectedRank,
                  inputTensor.shape());
        return Result::ERROR;
    }

    SignalAxes axes;
    if (MapSignalAxes(inputTensor,
                      IdentityAxisMap(inputTensor.rank()),
                      axes) != Result::SUCCESS) {
        JST_ERROR("[MODULE_DETECTOR] Input must contain valid signal axis metadata.");
        return Result::ERROR;
    }
    if (axes.sample && *axes.sample != kTimeAxis) {
        JST_ERROR("[MODULE_DETECTOR] Input sampleAxis must be absent or {} for [A, F, T, P] input.",
                  kTimeAxis);
        return Result::ERROR;
    }
    if (axes.channel && *axes.channel != kFrequencyAxis) {
        JST_ERROR("[MODULE_DETECTOR] Input channelAxis must be {} when present.",
                  kFrequencyAxis);
        return Result::ERROR;
    }
    if (axes.batch && *axes.batch != kAspectAxis) {
        JST_ERROR("[MODULE_DETECTOR] Input batchAxis must be absent or {} when present.",
                  kAspectAxis);
        return Result::ERROR;
    }

    axes.sample = kTimeAxis;
    axes.channel = kFrequencyAxis;

    const Index polarizationAxis = kPolarizationAxis;
    if (inputTensor.shape(polarizationAxis) != kExpectedInputPolarizations) {
        JST_ERROR("[MODULE_DETECTOR] Input polarization dimension must be {}, but received {}.",
                  kExpectedInputPolarizations,
                  inputTensor.shape(polarizationAxis));
        return Result::ERROR;
    }

    const Index sampleAxis = kTimeAxis;
    if (inputTensor.shape(sampleAxis) % config.integrationRate != 0) {
        JST_ERROR("[MODULE_DETECTOR] Input time dimension {} is not divisible by the integration rate {}.",
                  inputTensor.shape(sampleAxis),
                  config.integrationRate);
        return Result::ERROR;
    }

    Shape outputShape = inputTensor.shape();
    outputShape[sampleAxis] /= config.integrationRate;
    outputShape[polarizationAxis] = config.numberOfOutputPolarizations;

    U64 outputElementCount = 1;
    for (const U64 dimension : outputShape) {
        if (!detail::CheckedMultiply(outputElementCount,
                                     dimension,
                                     outputElementCount)) {
            JST_ERROR("[MODULE_DETECTOR] Output exceeds the supported layout range.");
            return Result::ERROR;
        }
    }

    U64 outputSizeBytes = 0;
    if (!detail::CheckedMultiply(outputElementCount,
                                 static_cast<U64>(DataTypeSize(DataType::F32)),
                                 outputSizeBytes)) {
        JST_ERROR("[MODULE_DETECTOR] Output exceeds the supported byte range.");
        return Result::ERROR;
    }

    validatedSignalAxes = axes;
    validatedOutputShape = std::move(outputShape);
    validatedSampleAxis = sampleAxis;
    validatedPolarizationAxis = polarizationAxis;
    validatedInputSampleCount =
        inputTensor.size() / inputTensor.shape(polarizationAxis);
    validatedOutputSizeBytes = outputSizeBytes;
    return Result::SUCCESS;
}

Result DetectorImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer"));
    JST_CHECK(defineInterfaceOutput("buffer"));

    return Result::SUCCESS;
}

Result DetectorImpl::create() {
    inputTensor = inputs().at("buffer").tensor;
    signalAxes = validatedSignalAxes;
    outputShape = validatedOutputShape;
    sampleAxis = validatedSampleAxis;
    polarizationAxis = validatedPolarizationAxis;
    inputSampleCount = validatedInputSampleCount;

    JST_CHECK(outputTensor.create(inputTensor.device(), DataType::F32, outputShape));
    JST_CHECK(outputTensor.propagateAttributes(inputTensor));
    JST_CHECK(SetSignalAxes(outputTensor, signalAxes));

    if (inputTensor.hasAttribute("sampleRate")) {
        const Tensor inputCopy = inputTensor;
        const F64 decimation = static_cast<F64>(integrationRate);
        JST_CHECK(outputTensor.setDerivedAttribute(
            "sampleRate",
            [inputCopy, decimation]() -> std::any {
                const std::any sampleRate = inputCopy.attribute("sampleRate");
                if (const auto* value = std::any_cast<F32>(&sampleRate)) {
                    return std::any(static_cast<F32>(*value / decimation));
                }
                if (const auto* value = std::any_cast<F64>(&sampleRate)) {
                    return std::any(*value / decimation);
                }
                return {};
            }));
    }

    outputs()["buffer"].produced(name(), "buffer", outputTensor);

    return Result::SUCCESS;
}

Result DetectorImpl::destroy() {
    inputTensor = {};
    outputTensor = {};
    signalAxes = {};
    outputShape.clear();
    sampleAxis = 0;
    polarizationAxis = 0;
    inputSampleCount = 0;

    return Result::SUCCESS;
}

Result DetectorImpl::reconfigure() {
    return Result::RECREATE;
}

}  // namespace Jetstream::Modules
