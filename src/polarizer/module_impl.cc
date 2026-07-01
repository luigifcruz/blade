#include "module_impl.hh"

namespace Jetstream::Modules {

Result PolarizerImpl::validate() {
    const auto& config = *candidate();

    if (inputPolarization != "xy") {
        JST_ERROR("[MODULE_POLARIZER] The input must be XY.");
        return Result::ERROR;
    }
    else {
        if (
            outputPolarization != "x" &&
            outputPolarization != "y" &&
            outputPolarization != "lr"
        ) {
            JST_ERROR("[MODULE_POLARIZER] Unsupported output polarization for input XY: {}. Expected [X, Y, LR].",
                      config.outputPolarization);
            return Result::ERROR;
        }
    }

    if (config.blockSize == 0) {
        JST_ERROR("[MODULE_POLARIZER] The CUDA block size must be positive.");
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

    return Result::SUCCESS;
}

Result PolarizerImpl::reconfigure() {
    return Result::RECREATE;
}

}  // namespace Jetstream::Modules
