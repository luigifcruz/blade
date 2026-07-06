#include "module_impl.hh"

namespace Jetstream::Modules {

Result PhasorImpl::validate() {
    return Result::SUCCESS;
}

Result PhasorImpl::define() {
    JST_CHECK(defineInterfaceInput("antennaPositions"));
    JST_CHECK(defineInterfaceInput("antennaCalibrations"));
    JST_CHECK(defineInterfaceInput("boresightCoordinates"));
    JST_CHECK(defineInterfaceInput("beamCoordinates"));
    JST_CHECK(defineInterfaceInput("julianDate"));
    JST_CHECK(defineInterfaceInput("dut1"));
    JST_CHECK(defineInterfaceOutput("delays"));
    JST_CHECK(defineInterfaceOutput("phasors"));

    return Result::SUCCESS;
}

Result PhasorImpl::create() {
    antennaPositionTensor = inputs().at("antennaPositions").tensor;
    antennaCalibrationTensor = inputs().at("antennaCalibrations").tensor;
    boresightCoordinateTensor = inputs().at("boresightCoordinates").tensor;
    beamCoordinateTensor = inputs().at("beamCoordinates").tensor;
    julianDateTensor = inputs().at("julianDate").tensor;
    dut1Tensor = inputs().at("dut1").tensor;

    if (antennaPositionTensor.rank() != 2) {
        JST_ERROR("[MODULE_PHASOR] Antenna Position expected to be rank 2.");
        return Result::ERROR;
    }
    if (antennaPositionTensor.shape()[1] != 3) {
        JST_ERROR("[MODULE_PHASOR] Antenna Position shape expected to be [:, 3].");
        return Result::ERROR;
    }
    if (!antennaPositionTensor.contiguous()) {
        JST_ERROR("[MODULE_PHASOR] Antenna Position tensor must be contiguous.");
        return Result::ERROR;
    }

    if (antennaCalibrationTensor.rank() != 3) {
        JST_ERROR("[MODULE_PHASOR] Antenna Calibration tensor expected to be rank 3 (AFP).");
        return Result::ERROR;
    }
    if (antennaCalibrationTensor.dtype() != DataType::CF64) {
        JST_ERROR("[MODULE_PHASOR] Antenna Calibration tensor data type expected to be CF64.");
        return Result::ERROR;
    }

    if (boresightCoordinateTensor.rank() != 1) {
        JST_ERROR("[MODULE_PHASOR] Boresight Coordinate tensor expected to be rank 1.");
        return Result::ERROR;
    }
    if (boresightCoordinateTensor.size() != 2) {
        JST_ERROR("[MODULE_PHASOR] Boresight Coordinate tensor expected to be of size 2.");
        return Result::ERROR;
    }

    if (beamCoordinateTensor.rank() != 2) {
        JST_ERROR("[MODULE_PHASOR] Beam Coordinate tensor expected to be rank 2.");
        return Result::ERROR;
    }
    Shape beamCoordShape = beamCoordinateTensor.shape();
    if (beamCoordShape[1] != 2) {
        JST_ERROR("[MODULE_PHASOR] Beam Coordinate tensor expected to be of shape [:, 2].");
        return Result::ERROR;
    }

    if (julianDateTensor.size() != 1) {
        JST_ERROR("[MODULE_PHASOR] Julian Date tensor expected to be of size 1.");
        return Result::ERROR;
    }

    if (dut1Tensor.size() != 1) {
        JST_ERROR("[MODULE_PHASOR] DUT1 tensor expected to be of size 1.");
        return Result::ERROR;
    }
    
    const auto& nofAntennas = antennaPositionTensor.shape()[0];
    if (antennaCalibrationTensor.shape()[0] != nofAntennas) {
        JST_ERROR(
            "[MODULE_PHASOR] Antenna Calibration shape does not match number of antenna postions: {}.",
            nofAntennas
        );
        return Result::ERROR;
    }
    const auto& nofBeams = beamCoordinateTensor.shape()[0];
    const auto& nofChannels = antennaCalibrationTensor.shape()[1];
    const auto& nofPolarizations = antennaCalibrationTensor.shape()[2];
    
    JST_CHECK(outputDelayTensor.create(
        DeviceType::CPU,
        DataType::F64,
        {nofBeams, nofAntennas}
    ));

    // TODO suppoprt frequency axis expansion by some rate
    JST_CHECK(outputPhasorTensor.create(
        DeviceType::CPU,
        DataType::CF32,
        {nofBeams, nofAntennas, nofChannels, nofPolarizations}
    ));
    
    outputs()["delays"].produced(name(), "delays", outputDelayTensor);
    outputs()["phasors"].produced(name(), "phasors", outputPhasorTensor);

    return Result::SUCCESS;
}

Result PhasorImpl::destroy() {
    antennaPositionTensor = {};
    antennaCalibrationTensor = {};
    boresightCoordinateTensor = {};
    beamCoordinateTensor = {};
    julianDateTensor = {};
    dut1Tensor = {};
    outputDelayTensor = {};
    outputPhasorTensor = {};

    return Result::SUCCESS;
}

Result PhasorImpl::reconfigure() {
    return Result::RECREATE;
}

}  // namespace Jetstream::Modules
