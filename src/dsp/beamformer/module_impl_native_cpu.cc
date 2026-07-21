#include <array>
#include <cmath>
#include <functional>
#include <type_traits>
#include <vector>

#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

namespace {

template<typename T>
CF32 ConvertSample(const T& value) {
    if constexpr (std::is_same_v<T, CI8>) {
        constexpr F32 kScale = 1.0f / 128.0f;
        return {
            static_cast<F32>(value.real()) * kScale,
            static_cast<F32>(value.imag()) * kScale,
        };
    } else {
        return value;
    }
}

CF32 Multiply(const CF32& lhs, const CF32& rhs) {
    return {
        (lhs.real() * rhs.real()) - (lhs.imag() * rhs.imag()),
        (lhs.real() * rhs.imag()) + (lhs.imag() * rhs.real()),
    };
}

}  // namespace

struct BeamformerImplNativeCpu : public BeamformerImpl,
                                 public NativeCpuRuntimeContext,
                                 public Scheduler::Context {
 public:
    Result create() final;
    Result destroy() override;
    Result computeSubmit() override;

 private:
    template<typename T>
    Result kernelTyped();

    std::function<Result()> kernel;
    std::vector<std::array<CF32, kExpectedPolarizations>> antennaCache;
};

Result BeamformerImplNativeCpu::create() {
    const Tensor& input = inputs().at("buffer").tensor;
    const Tensor& phasors = inputs().at("phasors").tensor;

    if (input.dtype() != DataType::CI8 && input.dtype() != DataType::CF32) {
        JST_ERROR("[MODULE_BEAMFORMER_NATIVE_CPU] Unsupported input data type '{}'. Expected CI8 or CF32.",
                  input.dtype());
        return Result::ERROR;
    }

    if (phasors.dtype() != DataType::CF32) {
        JST_ERROR("[MODULE_BEAMFORMER_NATIVE_CPU] Unsupported phasor data type '{}'. Expected CF32.",
                  phasors.dtype());
        return Result::ERROR;
    }

    JST_CHECK(BeamformerImpl::create());

    antennaCache.resize(inputTensor.shape(kBufferAspectAxis));
    if (inputTensor.dtype() == DataType::CI8) {
        kernel = [this]() { return kernelTyped<CI8>(); };
    } else {
        kernel = [this]() { return kernelTyped<CF32>(); };
    }

    return Result::SUCCESS;
}

Result BeamformerImplNativeCpu::destroy() {
    kernel = {};
    antennaCache.clear();
    return BeamformerImpl::destroy();
}

Result BeamformerImplNativeCpu::computeSubmit() {
    JST_CHECK(kernel());

    if (inputTensor.hasAttribute("timestamp")) {
        JST_CHECK(outputTensor.setAttribute("timestamp", inputTensor.attribute("timestamp")));
    }

    return Result::SUCCESS;
}

template<typename T>
Result BeamformerImplNativeCpu::kernelTyped() {
    const T* input = inputTensor.data<T>();
    const CF32* phasors = phasorTensor.data<CF32>();
    CF32* output = outputTensor.data<CF32>();

    const U64 antennaCount = inputTensor.shape(kBufferAspectAxis);
    const U64 frequencyCount = inputTensor.shape(kBufferFrequencyAxis);
    const U64 timeCount = inputTensor.shape(kBufferTimeAxis);

    for (U64 frequency = 0; frequency < frequencyCount; ++frequency) {
        for (U64 time = 0; time < timeCount; ++time) {
            std::array<F32, kExpectedPolarizations> incoherentPower{};

            for (U64 antenna = 0; antenna < antennaCount; ++antenna) {
                const U64 inputBase =
                    (((antenna * frequencyCount) + frequency) * timeCount + time) *
                    kExpectedPolarizations;
                const U64 phasorBase =
                    ((antenna * frequencyCount) + frequency) * kExpectedPolarizations;

                for (U64 polarization = 0; polarization < kExpectedPolarizations;
                     ++polarization) {
                    const CF32 sample = ConvertSample(input[inputBase + polarization]);
                    antennaCache[antenna][polarization] = sample;

                    if (enableIncoherentBeam) {
                        const CF32 weighted =
                            Multiply(sample, phasors[phasorBase + polarization]);
                        incoherentPower[polarization] +=
                            (weighted.real() * weighted.real()) +
                            (weighted.imag() * weighted.imag());
                    }
                }
            }

            for (U64 beam = 0; beam < beamCount; ++beam) {
                std::array<CF32, kExpectedPolarizations> accumulator{};

                for (U64 antenna = 0; antenna < antennaCount; ++antenna) {
                    const U64 phasorBase =
                        (((beam * antennaCount) + antenna) * frequencyCount + frequency) *
                        kExpectedPolarizations;

                    for (U64 polarization = 0; polarization < kExpectedPolarizations;
                         ++polarization) {
                        accumulator[polarization] +=
                            Multiply(antennaCache[antenna][polarization],
                                     phasors[phasorBase + polarization]);
                    }
                }

                const U64 outputBase =
                    (((beam * frequencyCount) + frequency) * timeCount + time) *
                    kExpectedPolarizations;
                for (U64 polarization = 0; polarization < kExpectedPolarizations;
                     ++polarization) {
                    output[outputBase + polarization] = accumulator[polarization];
                }
            }

            if (enableIncoherentBeam) {
                const U64 outputBase =
                    (((beamCount * frequencyCount) + frequency) * timeCount + time) *
                    kExpectedPolarizations;
                for (U64 polarization = 0; polarization < kExpectedPolarizations;
                     ++polarization) {
                    const F32 value = enableIncoherentBeamSqrt
                        ? std::sqrt(incoherentPower[polarization])
                        : incoherentPower[polarization];
                    output[outputBase + polarization] = {value, 0.0f};
                }
            }
        }
    }

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(BeamformerImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
