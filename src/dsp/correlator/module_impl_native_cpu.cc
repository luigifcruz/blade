#include <algorithm>
#include <array>
#include <functional>
#include <memory>
#include <vector>

#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

namespace {

template<typename T>
struct DualPolarizationSample {
    T xReal{};
    T xImag{};
    T yReal{};
    T yImag{};
};

template<typename T>
struct CorrelationSum {
    T real{};
    T imagConjugateB{};
};

template<typename T>
struct CorrelatorScratch {
    std::vector<DualPolarizationSample<T>> samples;
    std::vector<std::array<CorrelationSum<T>, kOutputPolarizations>> baselines;
};

template<typename T>
void Accumulate(CorrelationSum<T>& sum, T aReal, T aImag, T bReal, T bImag) {
    sum.real += (aReal * bReal) + (aImag * bImag);
    sum.imagConjugateB += (aImag * bReal) - (aReal * bImag);
}

template<typename Input, typename Calculation>
DualPolarizationSample<Calculation> ConvertSample(const Input* input) {
    return {
        static_cast<Calculation>(input[0].real()),
        static_cast<Calculation>(input[0].imag()),
        static_cast<Calculation>(input[1].real()),
        static_cast<Calculation>(input[1].imag()),
    };
}

}  // namespace

struct CorrelatorImplNativeCpu : public CorrelatorImpl,
                                 public NativeCpuRuntimeContext,
                                 public Scheduler::Context {
 public:
    Result create() final;
    Result destroy() override;
    Result computeSubmit() override;

 private:
    template<typename Input, typename Calculation>
    void bindKernel();

    template<typename Input, typename Calculation>
    Result kernelTyped(CorrelatorScratch<Calculation>& scratch);

    std::function<Result()> kernel;
};

Result CorrelatorImplNativeCpu::create() {
    const Tensor& input = inputs().at("buffer").tensor;
    if (input.dtype() != DataType::CI8 && input.dtype() != DataType::CF32) {
        JST_ERROR("[MODULE_CORRELATOR_NATIVE_CPU] Unsupported input data type '{}'. Expected CI8 or CF32.",
                  input.dtype());
        return Result::ERROR;
    }

    JST_CHECK(CorrelatorImpl::create());

    if (inputTensor.dtype() == DataType::CF32 && calculationMode == "integer") {
        JST_ERROR("[MODULE_CORRELATOR_NATIVE_CPU] Integer calculation mode is not supported for CF32 input. "
                  "Use single_precision_fp or double_precision_fp.");
        return Result::ERROR;
    }

    if (inputTensor.dtype() == DataType::CI8 &&
        calculationMode == "integer" &&
        inputTensor.shape(kTimeAxis) > 65535) {
        JST_ERROR("[MODULE_CORRELATOR_NATIVE_CPU] Integer calculation mode supports at most 65535 time samples.");
        return Result::ERROR;
    }

    if (inputTensor.dtype() == DataType::CI8) {
        if (calculationMode == "integer") {
            bindKernel<CI8, I32>();
        } else if (calculationMode == "single_precision_fp") {
            bindKernel<CI8, F32>();
        } else {
            bindKernel<CI8, F64>();
        }
    } else if (calculationMode == "single_precision_fp") {
        bindKernel<CF32, F32>();
    } else {
        bindKernel<CF32, F64>();
    }

    return Result::SUCCESS;
}

Result CorrelatorImplNativeCpu::destroy() {
    kernel = {};
    return CorrelatorImpl::destroy();
}

Result CorrelatorImplNativeCpu::computeSubmit() {
    if (integrationStep == 0) {
        CF32* output = outputTensor.data<CF32>();
        std::fill(output, output + outputTensor.size(), CF32{});
    }

    JST_CHECK(kernel());

    integrationStep = (integrationStep + 1) % integrationRate;
    if (integrationStep != 0) {
        return Result::SKIP;
    }

    if (inputTensor.hasAttribute("timestamp")) {
        JST_CHECK(outputTensor.setAttribute("timestamp", inputTensor.attribute("timestamp")));
    }

    return Result::SUCCESS;
}

template<typename Input, typename Calculation>
void CorrelatorImplNativeCpu::bindKernel() {
    auto scratch = std::make_shared<CorrelatorScratch<Calculation>>();
    scratch->samples.resize(inputTensor.shape(kAspectAxis));
    scratch->baselines.resize(baselineCount);
    kernel = [this, scratch]() { return kernelTyped<Input, Calculation>(*scratch); };
}

template<typename Input, typename Calculation>
Result CorrelatorImplNativeCpu::kernelTyped(CorrelatorScratch<Calculation>& scratch) {
    const Input* input = inputTensor.data<Input>();
    CF32* output = outputTensor.data<CF32>();
    const U64 antennaCount = inputTensor.shape(kAspectAxis);
    const U64 frequencyCount = inputTensor.shape(kFrequencyAxis);
    const U64 timeCount = inputTensor.shape(kTimeAxis);

    for (U64 frequency = 0; frequency < frequencyCount; ++frequency) {
        std::fill(scratch.baselines.begin(), scratch.baselines.end(),
                  std::array<CorrelationSum<Calculation>, kOutputPolarizations>{});

        for (U64 time = 0; time < timeCount; ++time) {
            for (U64 antenna = 0; antenna < antennaCount; ++antenna) {
                const U64 inputBase =
                    (((antenna * frequencyCount) + frequency) * timeCount + time) *
                    kExpectedInputPolarizations;
                scratch.samples[antenna] = ConvertSample<Input, Calculation>(input + inputBase);
            }

            U64 baseline = 0;
            for (U64 antennaA = 0; antennaA < antennaCount; ++antennaA) {
                const auto& a = scratch.samples[antennaA];
                for (U64 antennaB = antennaA; antennaB < antennaCount;
                     ++antennaB, ++baseline) {
                    const auto& b = scratch.samples[antennaB];
                    auto& sums = scratch.baselines[baseline];

                    Accumulate(sums[0], a.xReal, a.xImag, b.xReal, b.xImag);
                    Accumulate(sums[1], a.xReal, a.xImag, b.yReal, b.yImag);
                    Accumulate(sums[2], a.yReal, a.yImag, b.xReal, b.xImag);
                    Accumulate(sums[3], a.yReal, a.yImag, b.yReal, b.yImag);
                }
            }
        }

        for (U64 baseline = 0; baseline < baselineCount; ++baseline) {
            const U64 outputBase =
                ((baseline * frequencyCount) + frequency) * kOutputPolarizations;
            for (U64 product = 0; product < kOutputPolarizations; ++product) {
                const auto& sum = scratch.baselines[baseline][product];
                const F32 imaginary = static_cast<F32>(sum.imagConjugateB) *
                                      (conjugateAntennaIndex == 1 ? 1.0f : -1.0f);
                output[outputBase + product] +=
                    CF32{static_cast<F32>(sum.real), imaginary};
            }
        }
    }

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(CorrelatorImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
