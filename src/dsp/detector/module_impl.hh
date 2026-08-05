#ifndef BLADE_DETECTOR_MODULE_IMPL_HH
#define BLADE_DETECTOR_MODULE_IMPL_HH

#include <blade/detector/module.hh>
#include <jetstream/detail/module_impl.hh>
#include <jetstream/memory/axis.hh>

namespace Jetstream::Modules {

constexpr Index kExpectedRank = 4;
constexpr Index kAspectAxis = 0;
constexpr Index kFrequencyAxis = 1;
constexpr Index kTimeAxis = 2;
constexpr Index kPolarizationAxis = 3;
constexpr U64 kExpectedInputPolarizations = 2;

struct DetectorImpl : public Module::Impl, public DynamicConfig<Detector> {
    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;
    Result reconfigure() override;

 protected:
    SignalAxes validatedSignalAxes;
    Shape validatedOutputShape;
    Index validatedSampleAxis = 0;
    Index validatedPolarizationAxis = 0;
    U64 validatedInputSampleCount = 0;
    U64 validatedOutputSizeBytes = 0;

    Tensor inputTensor;
    Tensor outputTensor;
    SignalAxes signalAxes;
    Shape outputShape;
    Index sampleAxis = 0;
    Index polarizationAxis = 0;
    U64 inputSampleCount = 0;
};

}  // namespace Jetstream::Modules

#endif  // BLADE_DETECTOR_MODULE_IMPL_HH
