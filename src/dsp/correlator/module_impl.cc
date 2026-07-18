#include "module_impl.hh"

#include <limits>

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

    if (inputTensor.shape(kTimeAxis) == 0) {
        JST_ERROR("[MODULE_CORRELATOR] Input time dimension must be positive.");
        return Result::ERROR;
    }

    U64 baselineFactorA = inputTensor.shape(kAspectAxis);
    if (baselineFactorA == std::numeric_limits<U64>::max()) {
        JST_ERROR("[MODULE_CORRELATOR] Input aspect dimension is too large.");
        return Result::ERROR;
    }

    U64 baselineFactorB = baselineFactorA + 1;
    if ((baselineFactorA % 2) == 0) {
        baselineFactorA /= 2;
    } else {
        baselineFactorB /= 2;
    }
    if (baselineFactorA > std::numeric_limits<U64>::max() / baselineFactorB) {
        JST_ERROR("[MODULE_CORRELATOR] Baseline count is too large.");
        return Result::ERROR;
    }

    baselineCount = baselineFactorA * baselineFactorB;
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
    integrationStep = 0;

    return Result::SUCCESS;
}

Result CorrelatorImpl::reconfigure() {
    return Result::RECREATE;
}

}  // namespace Jetstream::Modules
