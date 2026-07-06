#ifndef BLADE_CORRELATOR_MODULE_IMPL_HH
#define BLADE_CORRELATOR_MODULE_IMPL_HH

#include <blade/correlator/module.hh>
#include <jetstream/detail/module_impl.hh>

namespace Jetstream::Modules {

constexpr U64 kExpectedRank = 4;
constexpr U64 kAspectAxis = 0;
constexpr U64 kFrequencyAxis = 1;
constexpr U64 kTimeAxis = 2;
constexpr U64 kPolarizationAxis = 3;
constexpr U64 kExpectedInputPolarizations = 2;
constexpr U64 kOutputPolarizations = 4;

struct CorrelatorImpl : public Module::Impl, public DynamicConfig<Correlator> {
    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;
    Result reconfigure() override;

 protected:
    Tensor inputTensor;
    Tensor outputTensor;
    U64 baselineCount = 0;
    U64 blockSizeX = 0;
    U64 blockSizeY = 0;
    U64 integrationStep = 0;
    bool optimizeTimeDomain = false;
    bool sharedMemoryEnabled = false;
};

}  // namespace Jetstream::Modules

#endif  // BLADE_CORRELATOR_MODULE_IMPL_HH
