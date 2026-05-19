#ifndef BLADE_DETECTOR_MODULE_IMPL_HH
#define BLADE_DETECTOR_MODULE_IMPL_HH

#include <blade/detector/module.hh>
#include <jetstream/detail/module_impl.hh>

namespace Jetstream::Modules {

constexpr U64 kExpectedRank = 4;
constexpr U64 kAspectAxis = 0;
constexpr U64 kFrequencyAxis = 1;
constexpr U64 kTimeAxis = 2;
constexpr U64 kPolarizationAxis = 3;
constexpr U64 kExpectedInputPolarizations = 2;

struct DetectorImpl : public Module::Impl, public DynamicConfig<Detector> {
    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;
    Result reconfigure() override;

 protected:
    Tensor inputTensor;
    Tensor outputTensor;
    U64 inputSampleCount = 0;
};

}  // namespace Jetstream::Modules

#endif  // BLADE_DETECTOR_MODULE_IMPL_HH
