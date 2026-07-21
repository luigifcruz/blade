#include "module_impl.hh"

#include <limits>

namespace Jetstream::Modules {

Result StackerImpl::validate() {
    const auto& config = *candidate();

    if (config.ratio == 0) {
        JST_ERROR("[MODULE_STACKER] The axis ratio must be greater than 0.");
        return Result::ERROR;
    }

    if (device() == DeviceType::CUDA && config.ratio != 1 &&
        (config.blockSize == 0 || config.blockSize > 1024)) {
        JST_ERROR("[MODULE_STACKER] The CUDA block size must be between 1 and 1024.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result StackerImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer"));
    JST_CHECK(defineInterfaceOutput("buffer"));

    return Result::SUCCESS;
}

Result StackerImpl::create() {
    inputTensor = inputs().at("buffer").tensor;

    if (inputTensor.rank() <= axis) {
        JST_ERROR("[MODULE_STACKER] Selected axis must exist on input.");
        return Result::ERROR;
    }

    if (!inputTensor.contiguous()) {
        JST_ERROR("[MODULE_STACKER] Input tensor must be contiguous.");
        return Result::ERROR;
    }

    bypass = ratio == 1;
    if (bypass) {
        outputTensor = inputTensor;
        outputs()["buffer"].produced(name(), "buffer", outputTensor);
        return Result::SUCCESS;
    }

    Shape outputShape = inputTensor.shape();
    if (outputShape[axis] != 0 &&
        ratio > std::numeric_limits<U64>::max() / outputShape[axis]) {
        JST_ERROR("[MODULE_STACKER] Output axis size is too large.");
        return Result::ERROR;
    }
    outputShape[axis] *= ratio;
    
    JST_CHECK(outputTensor.create(inputTensor.device(), inputTensor.dtype(), outputShape));
    JST_CHECK(outputTensor.propagateAttributes(inputTensor));

    outputs()["buffer"].produced(name(), "buffer", outputTensor);

    return Result::SUCCESS;
}

Result StackerImpl::destroy() {
    inputTensor = {};
    outputTensor = {};
    bypass = false;

    return Result::SUCCESS;
}

Result StackerImpl::reconfigure() {
    return Result::RECREATE;
}

}  // namespace Jetstream::Modules
