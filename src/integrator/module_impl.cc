#include "module_impl.hh"

namespace Jetstream::Modules {

Result IntegratorImpl::validate() {
    const auto& config = *candidate();

    if (config.rate < 1) {
        JST_ERROR("[MODULE_INTEGRATOR] The rate must be greater than 0.");
        return Result::ERROR;
    }
    if (config.size < 1) {
        JST_ERROR("[MODULE_INTEGRATOR] The size must be greater than 0.");
        return Result::ERROR;
    }
    if (config.rate*config.size == 1) {
        JST_ERROR("[MODULE_INTEGRATOR] Either size or rate must be greater than 1.");
        return Result::ERROR;
    }
    if (config.blockSize == 0) {
        JST_ERROR("[MODULE_INTEGRATOR] The CUDA block size must be positive.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result IntegratorImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer"));
    JST_CHECK(defineInterfaceOutput("buffer"));

    return Result::SUCCESS;
}

Result IntegratorImpl::create() {
    inputTensor = inputs().at("buffer").tensor;

    if (inputTensor.rank() <= axis) {
        JST_ERROR("[MODULE_INTEGRATOR] Selected axis must exist on input.");
        return Result::ERROR;
    }
    if (inputTensor.shape()[axis] % size != 0) {
        JST_ERROR(
            "[MODULE_INTEGRATOR] Selected axis must be a whole multiple of the integration size: {} % {} != 0.",
            inputTensor.shape()[axis], size
        );
        return Result::ERROR;
    }

    if (!inputTensor.contiguous()) {
        JST_ERROR("[MODULE_INTEGRATOR] Input tensor must be contiguous.");
        return Result::ERROR;
    }

    Shape outputShape = inputTensor.shape();
    outputShape[axis] /= size;
    
    JST_CHECK(outputTensor.create(inputTensor.device(), DataType::CF32, outputShape));
    JST_CHECK(outputTensor.propagateAttributes(inputTensor));

    outputs()["buffer"].produced(name(), "buffer", outputTensor);

    return Result::SUCCESS;
}

Result IntegratorImpl::destroy() {
    inputTensor = {};
    outputTensor = {};

    return Result::SUCCESS;
}

Result IntegratorImpl::reconfigure() {
    return Result::RECREATE;
}

}  // namespace Jetstream::Modules
