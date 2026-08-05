#ifndef BLADE_STACKER_MODULE_IMPL_HH
#define BLADE_STACKER_MODULE_IMPL_HH

#include <blade/stacker/module.hh>
#include <jetstream/detail/module_impl.hh>
#include <jetstream/memory/axis.hh>

namespace Jetstream::Modules {

struct StackerImpl : public Module::Impl, public DynamicConfig<Stacker> {
    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;
    Result reconfigure() override;

 protected:
    Tensor validatedInputTensor;
    Shape validatedOutputShape;
    SignalAxes validatedSignalAxes;
    bool validatedBypass = false;
    U64 validatedRatio = 1;
    U64 validatedWidth = 0;
    U64 validatedOutputWidth = 0;
    U64 validatedWidthByteSize = 0;
    U64 validatedOutputRowByteSize = 0;
    U64 validatedHeight = 0;
    U64 validatedInputSize = 0;
    U64 validatedOutputSizeBytes = 0;

    Tensor inputTensor;
    Tensor outputTensor;
    bool bypass = false;
    U64 stackRatio = 1;
    U64 width = 0;
    U64 outputWidth = 0;
    U64 widthByteSize = 0;
    U64 outputRowByteSize = 0;
    U64 height = 0;
    U64 inputSize = 0;
    U64 stackIndex;
};

}  // namespace Jetstream::Modules

#endif  // BLADE_STACKER_MODULE_IMPL_HH
