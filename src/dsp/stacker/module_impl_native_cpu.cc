#include <cstring>

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
    Result create() final;
    Result computeSubmit() override;

 private:
    U64 widthByteSize = 0;
    U64 height = 0;
    U64 stackIndex = 0;
};

Result StackerImplNativeCpu::create() {
    const Tensor& input = inputs().at("buffer").tensor;
    if (input.dtype() != DataType::F32 &&
        input.dtype() != DataType::CF32 &&
        input.dtype() != DataType::CI8) {
        JST_ERROR("[MODULE_STACKER_NATIVE_CPU] Unsupported input data type '{}'. Expected F32, CF32, or CI8.",
                  input.dtype());
        return Result::ERROR;
    }

    JST_CHECK(StackerImpl::create());

    stackIndex = 0;
    if (bypass) {
        return Result::SUCCESS;
    }

    U64 width = 1;
    for (U64 i = axis; i < inputTensor.rank(); ++i) {
        width *= inputTensor.shape(i);
    }
    widthByteSize = width * inputTensor.elementSize();

    height = 1;
    for (U64 i = 0; i < axis; ++i) {
        height *= inputTensor.shape(i);
    }

    return Result::SUCCESS;
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
            const U64 outputOffset = ((row * ratio) + stackIndex) * widthByteSize;
            std::memcpy(output + outputOffset, input + inputOffset, widthByteSize);
        }
    }

    if (inputTensor.hasAttribute("timestamp")) {
        JST_CHECK(outputTensor.setAttribute("timestamp", inputTensor.attribute("timestamp")));
    }

    stackIndex = (stackIndex + 1) % ratio;
    return stackIndex == 0 ? Result::SUCCESS : Result::SKIP;
}

JST_REGISTER_MODULE(StackerImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
