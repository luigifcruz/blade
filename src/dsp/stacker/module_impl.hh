#ifndef BLADE_STACKER_MODULE_IMPL_HH
#define BLADE_STACKER_MODULE_IMPL_HH

#include <blade/stacker/module.hh>
#include <jetstream/detail/module_impl.hh>

namespace Jetstream::Modules {

struct StackerImpl : public Module::Impl, public DynamicConfig<Stacker> {
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

#endif  // BLADE_STACKER_MODULE_IMPL_HH
