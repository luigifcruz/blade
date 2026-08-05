#ifndef BLADE_INTEGRATOR_MODULE_IMPL_HH
#define BLADE_INTEGRATOR_MODULE_IMPL_HH

#include <blade/integrator/module.hh>
#include <jetstream/detail/module_impl.hh>
#include <jetstream/memory/axis.hh>

namespace Jetstream::Modules {

struct IntegratorImpl : public Module::Impl, public DynamicConfig<Integrator> {
    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;
    Result reconfigure() override;

 protected:
    SignalAxes validatedSignalAxes;
    Shape validatedOutputShape;
    Index validatedResolvedAxis = 0;
    U64 validatedIntegratedElementCount = 0;
    U64 validatedNumberOfElements = 0;
    U64 validatedOutputSizeBytes = 0;
    bool validatedBypass = false;
    bool validatedAdjustSampleRate = false;

    Tensor inputTensor;
    Tensor outputTensor;
    SignalAxes signalAxes;
    Shape outputShape;
    Index resolvedAxis = 0;
    U64 integratedElementCount = 0;
    U64 numberOfElements = 0;
    U64 blockIndex;
    bool bypass = false;
    bool adjustSampleRate = false;
};

}  // namespace Jetstream::Modules

#endif  // BLADE_INTEGRATOR_MODULE_IMPL_HH
