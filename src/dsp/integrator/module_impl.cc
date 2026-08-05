#include "module_impl.hh"

#include <jetstream/tools/numeric.hh>

namespace Jetstream::Modules {

Result IntegratorImpl::validate() {
    validatedSignalAxes = {};
    validatedOutputShape.clear();
    validatedResolvedAxis = 0;
    validatedIntegratedElementCount = 0;
    validatedNumberOfElements = 0;
    validatedOutputSizeBytes = 0;
    validatedBypass = false;
    validatedAdjustSampleRate = false;

    const auto& config = *candidate();

    if (config.rate < 1) {
        JST_ERROR("[MODULE_INTEGRATOR] The rate must be greater than 0.");
        return Result::ERROR;
    }
    if (config.size < 1) {
        JST_ERROR("[MODULE_INTEGRATOR] The size must be greater than 0.");
        return Result::ERROR;
    }
    if (!inputs().contains("buffer")) {
        return Result::SUCCESS;
    }

    const Tensor& inputTensor = inputs().at("buffer").tensor;
    if (!inputTensor.validShape() || inputTensor.size() == 0) {
        return Result::SUCCESS;
    }

    SignalAxes outputAxes;
    if (MapSignalAxes(inputTensor,
                      IdentityAxisMap(inputTensor.rank()),
                      outputAxes) != Result::SUCCESS) {
        JST_ERROR("[MODULE_INTEGRATOR] Input contains invalid signal axis metadata.");
        return Result::ERROR;
    }

    if (config.axis >= inputTensor.rank()) {
        JST_ERROR("[MODULE_INTEGRATOR] Selected axis {} must exist on rank {} input.",
                  config.axis,
                  inputTensor.rank());
        return Result::ERROR;
    }
    if (inputTensor.shape(config.axis) % config.size != 0) {
        JST_ERROR(
            "[MODULE_INTEGRATOR] Selected axis must be a whole multiple of the integration size: {} % {} != 0.",
            inputTensor.shape(config.axis), config.size
        );
        return Result::ERROR;
    }

    Shape outputShape = inputTensor.shape();
    outputShape[config.axis] /= config.size;

    U64 integratedElementCount = 1;
    for (Index i = config.axis + 1; i < inputTensor.rank(); ++i) {
        if (!detail::CheckedMultiply(integratedElementCount,
                                     inputTensor.shape(i),
                                     integratedElementCount)) {
            JST_ERROR("[MODULE_INTEGRATOR] Integrated geometry exceeds the supported range.");
            return Result::ERROR;
        }
    }

    U64 outputElementCount = 1;
    for (const U64 dimension : outputShape) {
        if (!detail::CheckedMultiply(outputElementCount,
                                     dimension,
                                     outputElementCount)) {
            JST_ERROR("[MODULE_INTEGRATOR] Output exceeds the supported layout range.");
            return Result::ERROR;
        }
    }

    U64 outputSizeBytes = 0;
    if (!detail::CheckedMultiply(outputElementCount,
                                 static_cast<U64>(DataTypeSize(DataType::CF32)),
                                 outputSizeBytes)) {
        JST_ERROR("[MODULE_INTEGRATOR] Output exceeds the supported byte range.");
        return Result::ERROR;
    }

    const bool adjustSampleRate =
        outputAxes.sample && *outputAxes.sample == config.axis &&
        inputTensor.hasAttribute("sampleRate");
    validatedSignalAxes = outputAxes;
    validatedOutputShape = std::move(outputShape);
    validatedResolvedAxis = config.axis;
    validatedIntegratedElementCount = integratedElementCount;
    validatedNumberOfElements = outputElementCount;
    validatedOutputSizeBytes = outputSizeBytes;
    validatedAdjustSampleRate = adjustSampleRate;
    return Result::SUCCESS;
}

Result IntegratorImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer"));
    JST_CHECK(defineInterfaceOutput("buffer"));

    return Result::SUCCESS;
}

Result IntegratorImpl::create() {
    inputTensor = inputs().at("buffer").tensor;
    signalAxes = validatedSignalAxes;
    outputShape = validatedOutputShape;
    resolvedAxis = validatedResolvedAxis;
    integratedElementCount = validatedIntegratedElementCount;
    numberOfElements = validatedNumberOfElements;
    bypass = validatedBypass;
    adjustSampleRate = validatedAdjustSampleRate;
    blockIndex = 0;

    if (bypass) {
        outputTensor = inputTensor;
        outputs()["buffer"].produced(name(), "buffer", outputTensor);
        return Result::SUCCESS;
    }

    JST_CHECK(outputTensor.create(inputTensor.device(), DataType::CF32, outputShape));
    JST_CHECK(outputTensor.propagateAttributes(inputTensor));
    JST_CHECK(SetSignalAxes(outputTensor, signalAxes));

    if (adjustSampleRate) {
        const Tensor inputCopy = inputTensor;
        const F64 integrationSize = static_cast<F64>(size);
        JST_CHECK(outputTensor.setDerivedAttribute(
            "sampleRate",
            [inputCopy, integrationSize]() -> std::any {
                const std::any sampleRate = inputCopy.attribute("sampleRate");
                if (const auto* value = std::any_cast<F32>(&sampleRate)) {
                    return std::any(static_cast<F32>(*value / integrationSize));
                }
                if (const auto* value = std::any_cast<F64>(&sampleRate)) {
                    return std::any(*value / integrationSize);
                }
                return {};
            }));
    }

    outputs()["buffer"].produced(name(), "buffer", outputTensor);

    return Result::SUCCESS;
}

Result IntegratorImpl::destroy() {
    inputTensor = {};
    outputTensor = {};
    signalAxes = {};
    outputShape.clear();
    resolvedAxis = 0;
    integratedElementCount = 0;
    numberOfElements = 0;
    bypass = false;
    adjustSampleRate = false;
    blockIndex = 0;

    return Result::SUCCESS;
}

Result IntegratorImpl::reconfigure() {
    return Result::RECREATE;
}

}  // namespace Jetstream::Modules
