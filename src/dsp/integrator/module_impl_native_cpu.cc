#include <algorithm>
#include <functional>

#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

namespace {

CF32 ConvertToCF32(F32 value) {
    return {value, 0.0f};
}

CF32 ConvertToCF32(const CF32& value) {
    return value;
}

CF32 ConvertToCF32(const CI8& value) {
    return {static_cast<F32>(value.real()), static_cast<F32>(value.imag())};
}

}  // namespace

struct IntegratorImplNativeCpu : public IntegratorImpl,
                                 public NativeCpuRuntimeContext,
                                 public Scheduler::Context {
 public:
    Result create() final;
    Result computeSubmit() override;

 private:
    template<typename T>
    Result kernelTyped();

    std::function<Result()> kernel;
    U64 integratedElementCount = 1;
    U64 numberOfElements = 0;
    U64 blockIndex = 0;
};

Result IntegratorImplNativeCpu::create() {
    const Tensor& input = inputs().at("buffer").tensor;
    if (input.dtype() != DataType::F32 &&
        input.dtype() != DataType::CF32 &&
        input.dtype() != DataType::CI8) {
        JST_ERROR("[MODULE_INTEGRATOR_NATIVE_CPU] Unsupported input data type '{}'. Expected F32, CF32, or CI8.",
                  input.dtype());
        return Result::ERROR;
    }

    JST_CHECK(IntegratorImpl::create());

    blockIndex = 0;
    if (bypass) {
        return Result::SUCCESS;
    }

    integratedElementCount = 1;
    for (U64 i = axis + 1; i < inputTensor.rank(); ++i) {
        integratedElementCount *= inputTensor.shape(i);
    }
    numberOfElements = inputTensor.size() / size;

    switch (inputTensor.dtype()) {
        case DataType::F32:
            kernel = [this]() { return kernelTyped<F32>(); };
            break;
        case DataType::CF32:
            kernel = [this]() { return kernelTyped<CF32>(); };
            break;
        case DataType::CI8:
            kernel = [this]() { return kernelTyped<CI8>(); };
            break;
        default:
            return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result IntegratorImplNativeCpu::computeSubmit() {
    if (bypass) {
        return Result::SUCCESS;
    }

    CF32* output = outputTensor.data<CF32>();
    if (blockIndex == 0) {
        std::fill(output, output + numberOfElements, CF32{});
    }

    JST_CHECK(kernel());

    if (inputTensor.hasAttribute("timestamp")) {
        JST_CHECK(outputTensor.setAttribute("timestamp", inputTensor.attribute("timestamp")));
    }

    blockIndex = (blockIndex + 1) % rate;
    return blockIndex == 0 ? Result::SUCCESS : Result::SKIP;
}

template<typename T>
Result IntegratorImplNativeCpu::kernelTyped() {
    const T* input = inputTensor.data<T>();
    CF32* output = outputTensor.data<CF32>();
    const U64 groupCount = numberOfElements / integratedElementCount;

    for (U64 group = 0; group < groupCount; ++group) {
        const U64 inputGroupBase = group * size * integratedElementCount;
        const U64 outputGroupBase = group * integratedElementCount;

        for (U64 inner = 0; inner < integratedElementCount; ++inner) {
            CF32 accumulator{};
            for (U64 i = 0; i < size; ++i) {
                accumulator += ConvertToCF32(
                    input[inputGroupBase + (i * integratedElementCount) + inner]);
            }
            output[outputGroupBase + inner] += accumulator;
        }
    }

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(IntegratorImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
