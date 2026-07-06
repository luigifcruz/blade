#ifndef BLADE_BEAMFORMER_MODULE_IMPL_HH
#define BLADE_BEAMFORMER_MODULE_IMPL_HH

#include <blade/beamformer/module.hh>
#include <jetstream/detail/module_impl.hh>

namespace Jetstream::Modules {

constexpr U64 kBufferRank = 4;
constexpr U64 kPhasorRank = 5;

constexpr U64 kBufferAspectAxis = 0;
constexpr U64 kBufferFrequencyAxis = 1;
constexpr U64 kBufferTimeAxis = 2;
constexpr U64 kBufferPolarizationAxis = 3;

constexpr U64 kPhasorBeamAxis = 0;
constexpr U64 kPhasorAspectAxis = 1;
constexpr U64 kPhasorFrequencyAxis = 2;
constexpr U64 kPhasorTimeAxis = 3;
constexpr U64 kPhasorPolarizationAxis = 4;

constexpr U64 kExpectedPolarizations = 2;
constexpr U64 kExpectedPhasorTimeSamples = 1;

struct BeamformerImpl : public Module::Impl, public DynamicConfig<Beamformer> {
    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;
    Result reconfigure() override;

 protected:
    Tensor inputTensor;
    Tensor phasorTensor;
    Tensor outputTensor;
    U64 beamCount = 0;
};

}  // namespace Jetstream::Modules

#endif  // BLADE_BEAMFORMER_MODULE_IMPL_HH
