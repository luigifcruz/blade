#ifndef BLADE_PHASOR_MODULE_IMPL_HH
#define BLADE_PHASOR_MODULE_IMPL_HH

#include <blade/phasor/module.hh>
#include <jetstream/detail/module_impl.hh>

namespace Jetstream::Modules {

constexpr U64 kAntennaPositionRank = 2;
constexpr U64 kAntennaCalibrationRank = 3;
constexpr U64 kBoresightCoordinateRank = 1;
constexpr U64 kBeamCoordinateRank = 2;

constexpr U64 kAntennaAxis = 0;
constexpr U64 kCoordinateAxis = 1;
constexpr U64 kChannelAxis = 1;
constexpr U64 kPolarizationAxis = 2;

struct PhasorImpl : public Module::Impl, public DynamicConfig<Phasor> {
    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;
    Result reconfigure() override;

 protected:
    Tensor validatedAntennaPositionTensor;
    Tensor validatedAntennaCalibrationTensor;
    Tensor validatedBoresightCoordinateTensor;
    Tensor validatedBeamCoordinateTensor;
    Tensor validatedJulianDateTensor;
    Tensor validatedDut1Tensor;
    Shape validatedOutputDelayShape;
    Shape validatedOutputPhasorShape;
    U64 validatedAntennaCount = 0;
    U64 validatedOutputDelaySizeBytes = 0;
    U64 validatedOutputPhasorSizeBytes = 0;

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
