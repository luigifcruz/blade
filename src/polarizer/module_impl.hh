#ifndef BLADE_POLARIZER_MODULE_IMPL_HH
#define BLADE_POLARIZER_MODULE_IMPL_HH

#include <blade/polarizer/module.hh>
#include <jetstream/detail/module_impl.hh>

namespace Jetstream::Modules {

constexpr U64 kExpectedRank = 4;
constexpr U64 kAspectAxis = 0;
constexpr U64 kFrequencyAxis = 1;
constexpr U64 kTimeAxis = 2;
constexpr U64 kPolarizationAxis = 3;
constexpr U64 kExpectedInputPolarizations = 2;

struct PolarizerImpl : public Module::Impl, public DynamicConfig<Polarizer> {
    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;
    Result reconfigure() override;

 protected:
    Tensor inputTensor;
    Tensor outputTensor;
};

}  // namespace Jetstream::Modules

#endif  // BLADE_POLARIZER_MODULE_IMPL_HH
