#include "module_impl.hh"

#include <jetstream/tools/numeric.hh>

namespace Jetstream::Modules {

Result CorrelatorImpl::validate() {
    validatedBaselineCount = 0;
    validatedOutputSizeBytes = 0;
    validatedOutputShape.clear();
    validatedSignalAxes = {};

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

    if (!inputs().contains("buffer")) {
        return Result::SUCCESS;
    }

    const Tensor& input = inputs().at("buffer").tensor;
    if (!input.validShape() || input.size() == 0) {
        return Result::SUCCESS;
    }

    if (input.rank() != kExpectedRank) {
        JST_ERROR("[MODULE_CORRELATOR] Input tensor must have {} dimensions [A, F, T, P], but received shape {}.",
                  kExpectedRank,
                  input.shape());
        return Result::ERROR;
    }

    SignalAxes axes;
    if (MapSignalAxes(input,
                      IdentityAxisMap(input.rank()),
                      axes) != Result::SUCCESS) {
        JST_ERROR("[MODULE_CORRELATOR] Input must contain valid signal axis metadata.");
        return Result::ERROR;
    }

    if (axes.sample && *axes.sample != kTimeAxis) {
        JST_ERROR("[MODULE_CORRELATOR] sampleAxis must be absent or {}, but received {}.",
                  kTimeAxis,
                  *axes.sample);
        return Result::ERROR;
    }

    if (axes.channel && *axes.channel != kFrequencyAxis) {
        JST_ERROR("[MODULE_CORRELATOR] channelAxis must be absent or {}, but received {}.",
                  kFrequencyAxis,
                  *axes.channel);
        return Result::ERROR;
    }
    if (axes.batch && *axes.batch != kAspectAxis) {
        JST_ERROR("[MODULE_CORRELATOR] batchAxis must be absent or {}, but received {}.",
                  kAspectAxis,
                  *axes.batch);
        return Result::ERROR;
    }

    axes.sample = kTimeAxis;
    axes.channel = kFrequencyAxis;

    if (input.shape(kPolarizationAxis) != kExpectedInputPolarizations) {
        JST_ERROR("[MODULE_CORRELATOR] Input polarization dimension must be {}, but received {}.",
                  kExpectedInputPolarizations,
                  input.shape(kPolarizationAxis));
        return Result::ERROR;
    }

    const U64 antennaCount = input.shape(kAspectAxis);
    U64 baselineFactorB = 0;
    if (!detail::CheckedAdd(antennaCount, U64{1}, baselineFactorB)) {
        JST_ERROR("[MODULE_CORRELATOR] Input aspect dimension is too large.");
        return Result::ERROR;
    }

    U64 baselineFactorA = antennaCount;
    if ((baselineFactorA % 2) == 0) {
        baselineFactorA /= 2;
    } else {
        baselineFactorB /= 2;
    }

    U64 candidateBaselineCount = 0;
    if (!detail::CheckedMultiply(baselineFactorA,
                                 baselineFactorB,
                                 candidateBaselineCount)) {
        JST_ERROR("[MODULE_CORRELATOR] Baseline count is too large.");
        return Result::ERROR;
    }

    Shape outputShape = {
        candidateBaselineCount,
        input.shape(kFrequencyAxis),
        1,
        kOutputPolarizations,
    };

    U64 outputElementCount = 1;
    for (const U64 dimension : outputShape) {
        if (!detail::CheckedMultiply(outputElementCount,
                                     dimension,
                                     outputElementCount)) {
            JST_ERROR("[MODULE_CORRELATOR] Output shape exceeds the supported range.");
            return Result::ERROR;
        }
    }

    U64 outputSizeBytes = 0;
    if (!detail::CheckedMultiply(outputElementCount,
                                 static_cast<U64>(DataTypeSize(DataType::CF32)),
                                 outputSizeBytes)) {
        JST_ERROR("[MODULE_CORRELATOR] Output exceeds the supported byte range.");
        return Result::ERROR;
    }

    validatedBaselineCount = candidateBaselineCount;
    validatedOutputSizeBytes = outputSizeBytes;
    validatedOutputShape = std::move(outputShape);
    validatedSignalAxes = axes;

    return Result::SUCCESS;
}

Result CorrelatorImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer"));
    JST_CHECK(defineInterfaceOutput("buffer"));

    return Result::SUCCESS;
}

Result CorrelatorImpl::create() {
    inputTensor = inputs().at("buffer").tensor;
    baselineCount = validatedBaselineCount;

    JST_CHECK(outputTensor.create(inputTensor.device(), DataType::CF32, validatedOutputShape));
    JST_CHECK(outputTensor.propagateAttributes(inputTensor));
    JST_CHECK(SetSignalAxes(outputTensor, validatedSignalAxes));

    if (inputTensor.hasAttribute("sampleRate")) {
        const Tensor inputCopy = inputTensor;
        const F64 decimation = static_cast<F64>(inputTensor.shape(kTimeAxis)) *
                               static_cast<F64>(integrationRate);
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

    integrationStep = 0;

    return Result::SUCCESS;
}

Result CorrelatorImpl::destroy() {
    inputTensor = {};
    outputTensor = {};
    baselineCount = 0;
    integrationStep = 0;

    return Result::SUCCESS;
}

Result CorrelatorImpl::reconfigure() {
    return Result::RECREATE;
}

}  // namespace Jetstream::Modules
