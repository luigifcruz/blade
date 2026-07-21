#include <functional>

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
    Result create() final;
    Result computeSubmit() override;

 private:
    Result kernelXYtoLR();
    Result kernelXYtoLinear(U64 component);

    std::function<Result()> kernel;
};

Result PolarizerImplNativeCpu::create() {
    const Tensor& input = inputs().at("buffer").tensor;
    if (input.dtype() != DataType::CF32) {
        JST_ERROR("[MODULE_POLARIZER_NATIVE_CPU] Unsupported input data type '{}'. Expected CF32.",
                  input.dtype());
        return Result::ERROR;
    }

    JST_CHECK(PolarizerImpl::create());

    if (bypass) {
        return Result::SUCCESS;
    }

    if (outputPolarization == "lr") {
        kernel = [this]() { return kernelXYtoLR(); };
    } else if (outputPolarization == "x") {
        kernel = [this]() { return kernelXYtoLinear(0); };
    } else if (outputPolarization == "y") {
        kernel = [this]() { return kernelXYtoLinear(1); };
    } else {
        JST_ERROR("[MODULE_POLARIZER_NATIVE_CPU] Unsupported output polarization {}.",
                  outputPolarization);
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result PolarizerImplNativeCpu::computeSubmit() {
    if (bypass) {
        return Result::SUCCESS;
    }

    JST_CHECK(kernel());

    if (inputTensor.hasAttribute("timestamp")) {
        JST_CHECK(outputTensor.setAttribute("timestamp", inputTensor.attribute("timestamp")));
    }

    return Result::SUCCESS;
}

Result PolarizerImplNativeCpu::kernelXYtoLR() {
    const CF32* input = inputTensor.data<CF32>();
    CF32* output = outputTensor.data<CF32>();
    const U64 pairCount = inputTensor.size() / kExpectedInputPolarizations;

    for (U64 pair = 0; pair < pairCount; ++pair) {
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
