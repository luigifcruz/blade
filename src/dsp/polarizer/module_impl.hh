#ifndef BLADE_POLARIZER_MODULE_IMPL_HH
#define BLADE_POLARIZER_MODULE_IMPL_HH

#include <blade/polarizer/module.hh>
#include <jetstream/detail/module_impl.hh>
#include <jetstream/memory/axis.hh>

namespace Jetstream::Modules {

constexpr U64 kExpectedRank = 4;
constexpr U64 kAspectAxis = 0;
constexpr U64 kFrequencyAxis = 1;
constexpr U64 kTimeAxis = 2;
constexpr U64 kPolarizationAxis = 3;
constexpr U64 kExpectedInputPolarizations = 2;

enum class PolarizerPath : U8 {
    BYPASS,
    XY_TO_LR,
    XY_TO_X,
    XY_TO_Y,
};

struct PolarizerImpl : public Module::Impl, public DynamicConfig<Polarizer> {
    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;
    Result reconfigure() override;

 protected:
    Tensor validatedInputTensor;
    Shape validatedOutputShape;
    SignalAxes validatedSignalAxes;
    PolarizerPath validatedPath = PolarizerPath::BYPASS;
    U64 validatedOutputWorkItemCount = 0;
    U64 validatedOutputSizeBytes = 0;

    Tensor inputTensor;
    Tensor outputTensor;
    PolarizerPath path = PolarizerPath::BYPASS;
    bool bypass = false;
    U64 outputWorkItemCount = 0;
};

}  // namespace Jetstream::Modules

#endif  // BLADE_POLARIZER_MODULE_IMPL_HH
