#include "module_impl.hh"

namespace Jetstream::Modules {

Result PolarizerImpl::validate() {
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

    const bool sameBasis = config.inputPolarization == config.outputPolarization;
    if (!sameBasis &&
        (config.inputPolarization != "xy" ||
         (config.outputPolarization != "x" &&
          config.outputPolarization != "y" &&
          config.outputPolarization != "lr"))) {
        JST_ERROR("[MODULE_POLARIZER] Unsupported polarization conversion: {} -> {}.",
                  config.inputPolarization,
                  config.outputPolarization);
        return Result::ERROR;
    }

    if (config.blockSize == 0 || (!sameBasis && config.blockSize > 1024)) {
        JST_ERROR("[MODULE_POLARIZER] The CUDA block size must be between 1 and 1024.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result PolarizerImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer"));
    JST_CHECK(defineInterfaceOutput("buffer"));

    return Result::SUCCESS;
}

Result PolarizerImpl::create() {
    inputTensor = inputs().at("buffer").tensor;

    if (inputTensor.rank() != kExpectedRank) {
        JST_ERROR("[MODULE_POLARIZER] Input tensor must have {} dimensions [A, F, T, P], but received shape {}.",
                  kExpectedRank,
                  inputTensor.shape());
        return Result::ERROR;
    }

    if (!inputTensor.contiguous()) {
        JST_ERROR("[MODULE_POLARIZER] Input tensor must be contiguous.");
        return Result::ERROR;
    }

    bypass = inputPolarization == outputPolarization;
    if (bypass) {
        const U64 expectedPolarizations =
            (inputPolarization == "xy" || inputPolarization == "lr") ? 2 : 1;
        if (inputTensor.shape(kPolarizationAxis) != expectedPolarizations) {
            JST_ERROR("[MODULE_POLARIZER] Input polarization dimension must be {} for basis {}, but received {}.",
                      expectedPolarizations,
                      inputPolarization,
                      inputTensor.shape(kPolarizationAxis));
            return Result::ERROR;
        }

        outputTensor = inputTensor;
        outputs()["buffer"].produced(name(), "buffer", outputTensor);
        return Result::SUCCESS;
    }

    if (inputTensor.shape(kPolarizationAxis) != kExpectedInputPolarizations) {
        JST_ERROR("[MODULE_POLARIZER] Input polarization dimension must be {}, but received {}.",
                  kExpectedInputPolarizations,
                  inputTensor.shape(kPolarizationAxis));
        return Result::ERROR;
    }

    if (inputTensor.shape(kAspectAxis) == 0) {
        JST_ERROR("[MODULE_POLARIZER] Input aspect dimension must be positive.");
        return Result::ERROR;
    }

    Shape outputShape = inputTensor.shape();
    if (outputPolarization == "x" || outputPolarization == "y") {
        outputShape[kPolarizationAxis] = 1;
    }
    
    JST_CHECK(outputTensor.create(inputTensor.device(), DataType::CF32, outputShape));
    JST_CHECK(outputTensor.propagateAttributes(inputTensor));

    outputs()["buffer"].produced(name(), "buffer", outputTensor);

    return Result::SUCCESS;
}

Result PolarizerImpl::destroy() {
    inputTensor = {};
    outputTensor = {};
    bypass = false;

    return Result::SUCCESS;
}

Result PolarizerImpl::reconfigure() {
    return Result::RECREATE;
}

}  // namespace Jetstream::Modules
