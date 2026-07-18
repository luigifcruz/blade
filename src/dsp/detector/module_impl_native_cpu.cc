#include <functional>

#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct DetectorImplNativeCpu : public DetectorImpl,
                               public NativeCpuRuntimeContext,
                               public Scheduler::Context {
 public:
    Result create() final;
    Result computeSubmit() override;

 private:
    Result kernel1Pol();
    Result kernel4Pol();

    std::function<Result()> kernel;
};

Result DetectorImplNativeCpu::create() {
    const Tensor& input = inputs().at("buffer").tensor;
    if (input.dtype() != DataType::CF32) {
        JST_ERROR("[MODULE_DETECTOR_NATIVE_CPU] Unsupported input data type '{}'. Expected CF32.",
                  input.dtype());
        return Result::ERROR;
    }

    JST_CHECK(DetectorImpl::create());

    if (numberOfOutputPolarizations == 4) {
        kernel = [this]() { return kernel4Pol(); };
    } else if (numberOfOutputPolarizations == 1) {
        kernel = [this]() { return kernel1Pol(); };
    } else {
        JST_ERROR("[MODULE_DETECTOR_NATIVE_CPU] Unsupported number of output polarizations {}.",
                  numberOfOutputPolarizations);
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result DetectorImplNativeCpu::computeSubmit() {
    JST_CHECK(kernel());

    if (inputTensor.hasAttribute("timestamp")) {
        JST_CHECK(outputTensor.setAttribute("timestamp", inputTensor.attribute("timestamp")));
    }

    return Result::SUCCESS;
}

Result DetectorImplNativeCpu::kernel1Pol() {
    const CF32* input = inputTensor.data<CF32>();
    F32* output = outputTensor.data<F32>();
    const U64 outputSampleCount = inputSampleCount / integrationRate;

    for (U64 group = 0; group < outputSampleCount; ++group) {
        F32 totalPower = 0.0f;
        const U64 firstSample = group * integrationRate;

        for (U64 i = 0; i < integrationRate; ++i) {
            const U64 inputBase = (firstSample + i) * kExpectedInputPolarizations;
            const CF32& x = input[inputBase + 0];
            const CF32& y = input[inputBase + 1];
            totalPower += (x.real() * x.real()) + (x.imag() * x.imag()) +
                          (y.real() * y.real()) + (y.imag() * y.imag());
        }

        output[group] = totalPower;
    }

    return Result::SUCCESS;
}

Result DetectorImplNativeCpu::kernel4Pol() {
    const CF32* input = inputTensor.data<CF32>();
    F32* output = outputTensor.data<F32>();
    const U64 outputSampleCount = inputSampleCount / integrationRate;

    for (U64 group = 0; group < outputSampleCount; ++group) {
        F32 xx = 0.0f;
        F32 yy = 0.0f;
        F32 xyReal = 0.0f;
        F32 xyImag = 0.0f;
        const U64 firstSample = group * integrationRate;

        for (U64 i = 0; i < integrationRate; ++i) {
            const U64 inputBase = (firstSample + i) * kExpectedInputPolarizations;
            const CF32& x = input[inputBase + 0];
            const CF32& y = input[inputBase + 1];

            xx += (x.real() * x.real()) + (x.imag() * x.imag());
            yy += (y.real() * y.real()) + (y.imag() * y.imag());
            xyReal += (x.real() * y.real()) + (x.imag() * y.imag());
            xyImag += (x.imag() * y.real()) - (x.real() * y.imag());
        }

        const U64 outputBase = group * 4;
        output[outputBase + 0] = xx;
        output[outputBase + 1] = yy;
        output[outputBase + 2] = xyReal;
        output[outputBase + 3] = xyImag;
    }

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(DetectorImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
