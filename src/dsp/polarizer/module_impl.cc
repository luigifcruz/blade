#include "module_impl.hh"

#include <jetstream/tools/numeric.hh>

namespace Jetstream::Modules {

Result PolarizerImpl::validate() {
    validatedInputTensor = Tensor();
    validatedOutputShape.clear();
    validatedSignalAxes = {};
    validatedPath = PolarizerPath::BYPASS;
    validatedOutputWorkItemCount = 0;
    validatedOutputSizeBytes = 0;

    const auto& config = *candidate();

    const auto validPolarization = [](const std::string& polarization) {
        return polarization == "x" || polarization == "y" ||
               polarization == "l" || polarization == "r" ||
               polarization == "xy" || polarization == "lr";
    };

    if (!validPolarization(config.inputPolarization) ||
        !validPolarization(config.outputPolarization)) {
        JST_ERROR("[MODULE_POLARIZER] Unsupported polarization configuration: {} -> {}.",
                  config.inputPolarization,
                  config.outputPolarization);
        return Result::ERROR;
    }

    const bool candidateBypass =
        config.inputPolarization == config.outputPolarization;
    if (!candidateBypass &&
        (config.inputPolarization != "xy" ||
         (config.outputPolarization != "x" &&
          config.outputPolarization != "y" &&
          config.outputPolarization != "lr"))) {
        JST_ERROR("[MODULE_POLARIZER] Unsupported polarization conversion: {} -> {}.",
                  config.inputPolarization,
                  config.outputPolarization);
        return Result::ERROR;
    }

    PolarizerPath candidatePath = PolarizerPath::BYPASS;
    if (!candidateBypass) {
        if (config.outputPolarization == "lr") {
            candidatePath = PolarizerPath::XY_TO_LR;
        } else if (config.outputPolarization == "x") {
            candidatePath = PolarizerPath::XY_TO_X;
        } else {
            candidatePath = PolarizerPath::XY_TO_Y;
        }
    }
    validatedPath = candidatePath;

    if (!inputs().contains("buffer")) {
        return Result::SUCCESS;
    }

    const Tensor& input = inputs().at("buffer").tensor;
    if (!input.validShape() || input.size() == 0) {
        return Result::SUCCESS;
    }

    if (input.rank() != kExpectedRank) {
        JST_ERROR("[MODULE_POLARIZER] Input tensor must have {} dimensions [A, F, T, P], but received shape {}.",
                  kExpectedRank,
                  input.shape());
        return Result::ERROR;
    }

    SignalAxes signalAxes;
    if (MapSignalAxes(input,
                      IdentityAxisMap(input.rank()),
                      signalAxes) != Result::SUCCESS) {
        JST_ERROR("[MODULE_POLARIZER] Input signal axis metadata is invalid.");
        return Result::ERROR;
    }
    if (signalAxes.sample && *signalAxes.sample != kTimeAxis) {
        JST_ERROR("[MODULE_POLARIZER] Input sampleAxis must be absent or {}.", kTimeAxis);
        return Result::ERROR;
    }
    if (signalAxes.channel && *signalAxes.channel != kFrequencyAxis) {
        JST_ERROR("[MODULE_POLARIZER] Input channelAxis must be {} when present.",
                  kFrequencyAxis);
        return Result::ERROR;
    }
    if (signalAxes.batch && *signalAxes.batch != kAspectAxis) {
        JST_ERROR("[MODULE_POLARIZER] Input batchAxis must be absent or {}.",
                  kAspectAxis);
        return Result::ERROR;
    }

    signalAxes.sample = kTimeAxis;
    signalAxes.channel = kFrequencyAxis;

    const U64 expectedPolarizations = candidateBypass
        ? ((config.inputPolarization == "xy" || config.inputPolarization == "lr")
               ? 2
               : 1)
        : kExpectedInputPolarizations;
    if (input.shape(kPolarizationAxis) != expectedPolarizations) {
        JST_ERROR("[MODULE_POLARIZER] Input polarization dimension must be {} for basis {}, but received {}.",
                  expectedPolarizations,
                  config.inputPolarization,
                  input.shape(kPolarizationAxis));
        return Result::ERROR;
    }

    Shape outputShape = input.shape();
    if (candidatePath == PolarizerPath::XY_TO_X ||
        candidatePath == PolarizerPath::XY_TO_Y) {
        outputShape[kPolarizationAxis] = 1;
    }

    U64 outputElementCount = 1;
    for (const U64 dimension : outputShape) {
        if (!detail::CheckedMultiply(outputElementCount,
                                     dimension,
                                     outputElementCount)) {
            JST_ERROR("[MODULE_POLARIZER] Output shape exceeds the supported layout range.");
            return Result::ERROR;
        }
    }

    U64 outputSizeBytes = 0;
    if (!detail::CheckedMultiply(outputElementCount,
                                 static_cast<U64>(sizeof(CF32)),
                                 outputSizeBytes)) {
        JST_ERROR("[MODULE_POLARIZER] Output shape exceeds the supported byte range.");
        return Result::ERROR;
    }

    validatedInputTensor = input;
    validatedOutputShape = std::move(outputShape);
    validatedSignalAxes = signalAxes;
    validatedPath = candidatePath;
    validatedOutputWorkItemCount =
        outputElementCount / validatedOutputShape[kPolarizationAxis];
    validatedOutputSizeBytes = outputSizeBytes;

    return Result::SUCCESS;
}

Result PolarizerImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::STATELESS));

    JST_CHECK(defineInterfaceInput("buffer"));
    JST_CHECK(defineInterfaceOutput("buffer"));

    return Result::SUCCESS;
}

Result PolarizerImpl::create() {
    inputTensor = validatedInputTensor;
    path = validatedPath;
    bypass = path == PolarizerPath::BYPASS;
    outputWorkItemCount = validatedOutputWorkItemCount;

    if (bypass) {
        outputTensor = inputTensor;
        outputs()["buffer"].produced(name(), "buffer", outputTensor);
        return Result::SUCCESS;
    }

    JST_CHECK(outputTensor.create(inputTensor.device(),
                                  DataType::CF32,
                                  validatedOutputShape));
    JST_CHECK(outputTensor.propagateAttributes(inputTensor));
    JST_CHECK(SetSignalAxes(outputTensor, validatedSignalAxes));

    outputs()["buffer"].produced(name(), "buffer", outputTensor);

    return Result::SUCCESS;
}

Result PolarizerImpl::destroy() {
    inputTensor = {};
    outputTensor = {};
    path = PolarizerPath::BYPASS;
    bypass = false;
    outputWorkItemCount = 0;

    return Result::SUCCESS;
}

Result PolarizerImpl::reconfigure() {
    return Result::RECREATE;
}

}  // namespace Jetstream::Modules
