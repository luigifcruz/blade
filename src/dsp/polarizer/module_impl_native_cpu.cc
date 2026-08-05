#include <functional>
#include <limits>

#include <jetstream/memory/macros.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct PolarizerImplNativeCpu : public PolarizerImpl,
                                public NativeCpuRuntimeContext,
                                public Scheduler::Context {
 public:
    Result validate() final;
    Result create() final;
    Result computeSubmit() override;

 private:
    Result kernelXYtoLR();
    Result kernelXYtoLinear(U64 component);

    std::function<Result()> kernel;
};

Result PolarizerImplNativeCpu::validate() {
    JST_CHECK(PolarizerImpl::validate());

    if (!inputs().contains("buffer")) {
        return Result::SUCCESS;
    }

    const Tensor& input = inputs().at("buffer").tensor;
    if (!input.validShape() || input.size() == 0) {
        return Result::SUCCESS;
    }

    if (input.dtype() != DataType::CF32) {
        JST_ERROR("[MODULE_POLARIZER_NATIVE_CPU] Unsupported input data type '{}'. Expected CF32.",
                  input.dtype());
        return Result::ERROR;
    }

    if (validatedPath != PolarizerPath::BYPASS) {
        U64 alignedOutputSize = 0;
        if (!detail::CheckedPageAlignedSize(validatedOutputSizeBytes,
                                             alignedOutputSize) ||
            alignedOutputSize > std::numeric_limits<std::size_t>::max()) {
            JST_ERROR("[MODULE_POLARIZER_NATIVE_CPU] Output allocation size is too large.");
            return Result::ERROR;
        }
    }

    return Result::SUCCESS;
}

Result PolarizerImplNativeCpu::create() {
    kernel = {};
    JST_CHECK(PolarizerImpl::create());

    switch (path) {
        case PolarizerPath::BYPASS:
            break;
        case PolarizerPath::XY_TO_LR:
            kernel = [this]() { return kernelXYtoLR(); };
            break;
        case PolarizerPath::XY_TO_X:
            kernel = [this]() { return kernelXYtoLinear(0); };
            break;
        case PolarizerPath::XY_TO_Y:
            kernel = [this]() { return kernelXYtoLinear(1); };
            break;
    }

    return Result::SUCCESS;
}

Result PolarizerImplNativeCpu::computeSubmit() {
    if (bypass) {
        return Result::SUCCESS;
    }

    JST_CHECK(kernel());

    return Result::SUCCESS;
}

Result PolarizerImplNativeCpu::kernelXYtoLR() {
    const CF32* input = inputTensor.data<CF32>();
    CF32* output = outputTensor.data<CF32>();

    for (U64 pair = 0; pair < outputWorkItemCount; ++pair) {
        const CF32& x = input[(pair * 2) + 0];
        const CF32& y = input[(pair * 2) + 1];
        const CF32 y90{-y.imag(), y.real()};
        output[(pair * 2) + 0] = x + y90;
        output[(pair * 2) + 1] = x - y90;
    }

    return Result::SUCCESS;
}

Result PolarizerImplNativeCpu::kernelXYtoLinear(U64 component) {
    const CF32* input = inputTensor.data<CF32>();
    CF32* output = outputTensor.data<CF32>();

    for (U64 i = 0; i < outputTensor.size(); ++i) {
        output[i] = input[(i * 2) + component];
    }

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(PolarizerImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
