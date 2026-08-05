#include "module_impl.hh"

#include <jetstream/tools/numeric.hh>

namespace Jetstream::Modules {

Result StackerImpl::validate() {
    validatedInputTensor = Tensor();
    validatedOutputShape.clear();
    validatedSignalAxes = {};
    validatedBypass = false;
    validatedRatio = 1;
    validatedWidth = 0;
    validatedOutputWidth = 0;
    validatedWidthByteSize = 0;
    validatedOutputRowByteSize = 0;
    validatedHeight = 0;
    validatedInputSize = 0;
    validatedOutputSizeBytes = 0;

    const auto& config = *candidate();

    if (config.ratio == 0) {
        JST_ERROR("[MODULE_STACKER] The axis ratio must be greater than 0.");
        return Result::ERROR;
    }

    if (!inputs().contains("buffer")) {
        return Result::SUCCESS;
    }

    const Tensor& input = inputs().at("buffer").tensor;
    if (!input.validShape() || input.size() == 0) {
        return Result::SUCCESS;
    }

    if (config.axis >= input.rank()) {
        JST_ERROR("[MODULE_STACKER] Selected axis must exist on input.");
        return Result::ERROR;
    }

    SignalAxes signalAxes;
    if (MapSignalAxes(input,
                      IdentityAxisMap(input.rank()),
                      signalAxes) != Result::SUCCESS) {
        JST_ERROR("[MODULE_STACKER] Input signal axis metadata is invalid.");
        return Result::ERROR;
    }

    Shape outputShape = input.shape();
    if (!detail::CheckedMultiply(outputShape[config.axis],
                                 config.ratio,
                                 outputShape[config.axis])) {
        JST_ERROR("[MODULE_STACKER] Output axis size is too large.");
        return Result::ERROR;
    }

    U64 width = 1;
    for (Index axis = config.axis; axis < input.rank(); ++axis) {
        if (!detail::CheckedMultiply(width, input.shape(axis), width)) {
            JST_ERROR("[MODULE_STACKER] Input row width exceeds the supported range.");
            return Result::ERROR;
        }
    }

    U64 height = 1;
    for (Index axis = 0; axis < config.axis; ++axis) {
        if (!detail::CheckedMultiply(height, input.shape(axis), height)) {
            JST_ERROR("[MODULE_STACKER] Input row count exceeds the supported range.");
            return Result::ERROR;
        }
    }

    U64 outputWidth = 0;
    U64 widthByteSize = 0;
    U64 outputRowByteSize = 0;
    if (!detail::CheckedMultiply(width, config.ratio, outputWidth) ||
        !detail::CheckedMultiply(width,
                                 input.elementSize(),
                                 widthByteSize) ||
        !detail::CheckedMultiply(outputWidth,
                                 input.elementSize(),
                                 outputRowByteSize)) {
        JST_ERROR("[MODULE_STACKER] Stacked row size exceeds the supported range.");
        return Result::ERROR;
    }

    U64 outputElementCount = 1;
    for (const U64 dimension : outputShape) {
        if (!detail::CheckedMultiply(outputElementCount,
                                     dimension,
                                     outputElementCount)) {
            JST_ERROR("[MODULE_STACKER] Output shape exceeds the supported layout range.");
            return Result::ERROR;
        }
    }

    U64 outputSizeBytes = 0;
    if (!detail::CheckedMultiply(outputElementCount,
                                 input.elementSize(),
                                 outputSizeBytes)) {
        JST_ERROR("[MODULE_STACKER] Output shape exceeds the supported byte range.");
        return Result::ERROR;
    }

    validatedInputTensor = input;
    validatedOutputShape = std::move(outputShape);
    validatedSignalAxes = signalAxes;
    validatedBypass = config.ratio == 1;
    validatedRatio = config.ratio;
    validatedWidth = width;
    validatedOutputWidth = outputWidth;
    validatedWidthByteSize = widthByteSize;
    validatedOutputRowByteSize = outputRowByteSize;
    validatedHeight = height;
    validatedInputSize = input.size();
    validatedOutputSizeBytes = outputSizeBytes;

    return Result::SUCCESS;
}

Result StackerImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer"));
    JST_CHECK(defineInterfaceOutput("buffer"));

    return Result::SUCCESS;
}

Result StackerImpl::create() {
    inputTensor = validatedInputTensor;
    bypass = validatedBypass;
    stackRatio = validatedRatio;
    width = validatedWidth;
    outputWidth = validatedOutputWidth;
    widthByteSize = validatedWidthByteSize;
    outputRowByteSize = validatedOutputRowByteSize;
    height = validatedHeight;
    inputSize = validatedInputSize;
    stackIndex = 0;

    if (bypass) {
        outputTensor = inputTensor;
        outputs()["buffer"].produced(name(), "buffer", outputTensor);
        return Result::SUCCESS;
    }

    JST_CHECK(outputTensor.create(inputTensor.device(),
                                  inputTensor.dtype(),
                                  validatedOutputShape));
    JST_CHECK(outputTensor.propagateAttributes(inputTensor));
    JST_CHECK(SetSignalAxes(outputTensor, validatedSignalAxes));

    outputs()["buffer"].produced(name(), "buffer", outputTensor);

    return Result::SUCCESS;
}

Result StackerImpl::destroy() {
    inputTensor = {};
    outputTensor = {};
    bypass = false;
    stackRatio = 1;
    width = 0;
    outputWidth = 0;
    widthByteSize = 0;
    outputRowByteSize = 0;
    height = 0;
    inputSize = 0;

    return Result::SUCCESS;
}

Result StackerImpl::reconfigure() {
    return Result::RECREATE;
}

}  // namespace Jetstream::Modules
