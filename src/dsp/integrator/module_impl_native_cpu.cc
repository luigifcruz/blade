#include <algorithm>
#include <functional>
#include <limits>

#include <jetstream/memory/macros.hh>
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
    Result validate() final;
    Result create() final;
    Result computeSubmit() override;

 private:
    template<typename T>
    Result kernelTyped();

    std::function<Result()> kernel;
};

Result IntegratorImplNativeCpu::validate() {
    JST_CHECK(IntegratorImpl::validate());

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
        JST_ERROR("[MODULE_INTEGRATOR_NATIVE_CPU] Unsupported input data type '{}'. Expected F32, CF32, or CI8.",
                  input.dtype());
        return Result::ERROR;
    }

    const auto& config = *candidate();
    validatedBypass = config.size == 1 && config.rate == 1 &&
                      input.dtype() == DataType::CF32;
    if (!validatedBypass) {
        U64 alignedOutputSize = 0;
        if (!detail::CheckedPageAlignedSize(validatedOutputSizeBytes,
                                            alignedOutputSize) ||
            alignedOutputSize > std::numeric_limits<std::size_t>::max()) {
            JST_ERROR("[MODULE_INTEGRATOR_NATIVE_CPU] Output allocation size is too large.");
            return Result::ERROR;
        }
    }

    return Result::SUCCESS;
}

Result IntegratorImplNativeCpu::create() {
    JST_CHECK(IntegratorImpl::create());

    if (bypass) {
        return Result::SUCCESS;
    }

    if (inputTensor.dtype() == DataType::F32) {
        kernel = [this]() { return kernelTyped<F32>(); };
    } else if (inputTensor.dtype() == DataType::CF32) {
        kernel = [this]() { return kernelTyped<CF32>(); };
    } else {
        kernel = [this]() { return kernelTyped<CI8>(); };
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
