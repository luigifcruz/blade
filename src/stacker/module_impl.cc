#include "module_impl.hh"

namespace Jetstream::Modules {

Result StackerImpl::validate() {
    const auto& config = *candidate();

    if (config.ratio <= 1) {
        JST_ERROR("[MODULE_STACKER] The axis ratio must be greater than 1.");
        return Result::ERROR;
    }

    if (config.blockSize == 0) {
        JST_ERROR("[MODULE_STACKER] The CUDA block size must be positive.");
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

    Shape outputShape = inputTensor.shape();
    outputShape[axis] *= ratio;
    
    JST_CHECK(outputTensor.create(inputTensor.device(), DataType::CF32, outputShape));
    JST_CHECK(outputTensor.propagateAttributes(inputTensor));

    outputs()["buffer"].produced(name(), "buffer", outputTensor);

    return Result::SUCCESS;
}

Result StackerImpl::destroy() {
    inputTensor = {};
    outputTensor = {};

    return Result::SUCCESS;
}

Result StackerImpl::reconfigure() {
    return Result::RECREATE;
}

}  // namespace Jetstream::Modules
