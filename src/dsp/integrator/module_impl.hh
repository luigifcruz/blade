#ifndef BLADE_INTEGRATOR_MODULE_IMPL_HH
#define BLADE_INTEGRATOR_MODULE_IMPL_HH

#include <blade/integrator/module.hh>
#include <jetstream/detail/module_impl.hh>

namespace Jetstream::Modules {

struct IntegratorImpl : public Module::Impl, public DynamicConfig<Integrator> {
    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;
    Result reconfigure() override;

 protected:
    Tensor inputTensor;
    Tensor outputTensor;
    bool bypass = false;
};

}  // namespace Jetstream::Modules

#endif  // BLADE_INTEGRATOR_MODULE_IMPL_HH
