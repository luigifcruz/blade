#include <cstring>
#include <limits>

#include <jetstream/memory/macros.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct StackerImplNativeCpu : public StackerImpl,
                              public NativeCpuRuntimeContext,
                              public Scheduler::Context {
 public:
    Result validate() final;
    Result create() final;
    Result computeSubmit() override;
};

Result StackerImplNativeCpu::validate() {
    JST_CHECK(StackerImpl::validate());

    if (!inputs().contains("buffer")) {
        return Result::SUCCESS;
    }

    const Tensor& input = inputs().at("buffer").tensor;
    if (!input.validShape() || input.size() == 0) {
        return Result::SUCCESS;
    }

    if (input.dtype() != DataType::F32 &&
        input.dtype() != DataType::CF32 &&
        input.dtype() != DataType::CI8) {
        JST_ERROR("[MODULE_STACKER_NATIVE_CPU] Unsupported input data type '{}'. Expected F32, CF32, or CI8.",
                  input.dtype());
        return Result::ERROR;
    }

    if (!validatedBypass) {
        U64 alignedOutputSize = 0;
        if (!detail::CheckedPageAlignedSize(validatedOutputSizeBytes,
                                             alignedOutputSize) ||
            alignedOutputSize > std::numeric_limits<std::size_t>::max()) {
            JST_ERROR("[MODULE_STACKER_NATIVE_CPU] Output allocation size is too large.");
            return Result::ERROR;
        }
    }

    return Result::SUCCESS;
}

Result StackerImplNativeCpu::create() {
    return StackerImpl::create();
}

Result StackerImplNativeCpu::computeSubmit() {
    if (bypass) {
        return Result::SUCCESS;
    }

    U8* output = outputTensor.data<U8>();
    const U8* input = inputTensor.data<U8>();

    if (stackIndex == 0 && outputTensor.sizeBytes() != 0) {
        std::memset(output, 0, outputTensor.sizeBytes());
    }

    if (widthByteSize != 0) {
        for (U64 row = 0; row < height; ++row) {
            const U64 inputOffset = row * widthByteSize;
            const U64 outputOffset = (row * outputRowByteSize) +
                                     (stackIndex * widthByteSize);
            std::memcpy(output + outputOffset, input + inputOffset, widthByteSize);
        }
    }

    stackIndex = (stackIndex + 1) % stackRatio;
    return stackIndex == 0 ? Result::SUCCESS : Result::SKIP;
}

JST_REGISTER_MODULE(StackerImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
