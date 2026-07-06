#ifndef BLADE_PHASOR_MODULE_IMPL_HH
#define BLADE_PHASOR_MODULE_IMPL_HH

#include <blade/phasor/module.hh>
#include <jetstream/detail/module_impl.hh>

namespace Jetstream::Modules {

struct PhasorImpl : public Module::Impl, public DynamicConfig<Phasor> {
    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;
    Result reconfigure() override;

 protected:
    Tensor antennaPositionTensor;
    Tensor antennaCalibrationTensor;
    Tensor boresightCoordinateTensor;
    Tensor beamCoordinateTensor;
    Tensor julianDateTensor;
    Tensor dut1Tensor;
    Tensor outputDelayTensor;
    Tensor outputPhasorTensor;
};

}  // namespace Jetstream::Modules

#endif  // BLADE_PHASOR_MODULE_IMPL_HH
