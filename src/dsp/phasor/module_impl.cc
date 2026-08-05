#include "module_impl.hh"

#include <cmath>

#include <jetstream/tools/numeric.hh>

namespace Jetstream::Modules {

Result PhasorImpl::validate() {
    validatedAntennaPositionTensor = {};
    validatedAntennaCalibrationTensor = {};
    validatedBoresightCoordinateTensor = {};
    validatedBeamCoordinateTensor = {};
    validatedJulianDateTensor = {};
    validatedDut1Tensor = {};
    validatedOutputDelayShape.clear();
    validatedOutputPhasorShape.clear();
    validatedAntennaCount = 0;
    validatedOutputDelaySizeBytes = 0;
    validatedOutputPhasorSizeBytes = 0;

    const auto& config = *candidate();
    if (!std::isfinite(config.observationFrequencyHz) ||
        !std::isfinite(config.channelBandwidthHz) ||
        !std::isfinite(config.totalBandwidthHz)) {
        JST_ERROR("[MODULE_PHASOR] Frequency configuration must be finite.");
        return Result::ERROR;
    }
    if (!std::isfinite(config.arrayReferenceLongitude) ||
        !std::isfinite(config.arrayReferenceLatitude) ||
        !std::isfinite(config.arrayReferenceAltitude)) {
        JST_ERROR("[MODULE_PHASOR] Array reference configuration must be finite.");
        return Result::ERROR;
    }
    if (!std::isfinite(config.observationFrequencyHz -
                       (config.totalBandwidthHz / 2.0))) {
        JST_ERROR("[MODULE_PHASOR] Band-start frequency exceeds the supported range.");
        return Result::ERROR;
    }

    if (!inputs().contains("antennaPositions") ||
        !inputs().contains("antennaCalibrations") ||
        !inputs().contains("boresightCoordinates") ||
        !inputs().contains("beamCoordinates") ||
        !inputs().contains("julianDate") ||
        !inputs().contains("dut1")) {
        return Result::SUCCESS;
    }

    const Tensor& antennaPositions = inputs().at("antennaPositions").tensor;
    const Tensor& antennaCalibrations = inputs().at("antennaCalibrations").tensor;
    const Tensor& boresightCoordinates = inputs().at("boresightCoordinates").tensor;
    const Tensor& beamCoordinates = inputs().at("beamCoordinates").tensor;
    const Tensor& julianDate = inputs().at("julianDate").tensor;
    const Tensor& dut1 = inputs().at("dut1").tensor;
    if (!antennaPositions.validShape() || antennaPositions.size() == 0 ||
        !antennaCalibrations.validShape() || antennaCalibrations.size() == 0 ||
        !boresightCoordinates.validShape() || boresightCoordinates.size() == 0 ||
        !beamCoordinates.validShape() || beamCoordinates.size() == 0 ||
        !julianDate.validShape() || julianDate.size() == 0 ||
        !dut1.validShape() || dut1.size() == 0) {
        return Result::SUCCESS;
    }

    if (antennaPositions.rank() != kAntennaPositionRank) {
        JST_ERROR("[MODULE_PHASOR] Antenna Position expected to be rank 2.");
        return Result::ERROR;
    }
    if (antennaPositions.shape(kCoordinateAxis) != 3) {
        JST_ERROR("[MODULE_PHASOR] Antenna Position shape expected to be [:, 3].");
        return Result::ERROR;
    }

    if (antennaCalibrations.rank() != kAntennaCalibrationRank) {
        JST_ERROR("[MODULE_PHASOR] Antenna Calibration tensor expected to be rank 3 (AFP).");
        return Result::ERROR;
    }

    if (boresightCoordinates.rank() != kBoresightCoordinateRank) {
        JST_ERROR("[MODULE_PHASOR] Boresight Coordinate tensor expected to be rank 1.");
        return Result::ERROR;
    }
    if (boresightCoordinates.size() != 2) {
        JST_ERROR("[MODULE_PHASOR] Boresight Coordinate tensor expected to be of size 2.");
        return Result::ERROR;
    }

    if (beamCoordinates.rank() != kBeamCoordinateRank) {
        JST_ERROR("[MODULE_PHASOR] Beam Coordinate tensor expected to be rank 2.");
        return Result::ERROR;
    }
    if (beamCoordinates.shape(kCoordinateAxis) != 2) {
        JST_ERROR("[MODULE_PHASOR] Beam Coordinate tensor expected to be of shape [:, 2].");
        return Result::ERROR;
    }

    if (julianDate.size() != 1) {
        JST_ERROR("[MODULE_PHASOR] Julian Date tensor expected to be of size 1.");
        return Result::ERROR;
    }
    if (dut1.size() != 1) {
        JST_ERROR("[MODULE_PHASOR] DUT1 tensor expected to be of size 1.");
        return Result::ERROR;
    }

    const U64 antennaCount = antennaPositions.shape(kAntennaAxis);
    if (antennaCalibrations.shape(kAntennaAxis) != antennaCount) {
        JST_ERROR("[MODULE_PHASOR] Antenna Calibration shape does not match number of antenna positions: {}.",
                  antennaCount);
        return Result::ERROR;
    }
    if (config.referenceAntennaIndex >= antennaCount) {
        JST_ERROR("[MODULE_PHASOR] Reference antenna index {} is out of range for {} antennas.",
                  config.referenceAntennaIndex,
                  antennaCount);
        return Result::ERROR;
    }

    const U64 beamCount = beamCoordinates.shape(kAntennaAxis);
    const U64 channelCount = antennaCalibrations.shape(kChannelAxis);
    const U64 polarizationCount = antennaCalibrations.shape(kPolarizationAxis);
    U64 lastFrequencyIndex = 0;
    if (!detail::CheckedAdd(config.frequencyStartIndex,
                            channelCount - 1,
                            lastFrequencyIndex)) {
        JST_ERROR("[MODULE_PHASOR] Frequency channel index exceeds the supported range.");
        return Result::ERROR;
    }
    if (!std::isfinite(static_cast<F64>(lastFrequencyIndex) *
                       config.channelBandwidthHz)) {
        JST_ERROR("[MODULE_PHASOR] Channel frequency exceeds the supported range.");
        return Result::ERROR;
    }

    Shape outputDelayShape = {beamCount, antennaCount};
    U64 outputDelayElementCount = 0;
    U64 outputDelaySizeBytes = 0;
    if (!detail::CheckedMultiply(beamCount,
                                 antennaCount,
                                 outputDelayElementCount) ||
        !detail::CheckedMultiply(outputDelayElementCount,
                                 static_cast<U64>(DataTypeSize(DataType::F64)),
                                 outputDelaySizeBytes)) {
        JST_ERROR("[MODULE_PHASOR] Delay output shape exceeds the supported range.");
        return Result::ERROR;
    }

    Shape outputPhasorShape = {
        beamCount,
        antennaCount,
        channelCount,
        polarizationCount,
    };
    U64 outputPhasorElementCount = 1;
    for (const U64 dimension : outputPhasorShape) {
        if (!detail::CheckedMultiply(outputPhasorElementCount,
                                     dimension,
                                     outputPhasorElementCount)) {
            JST_ERROR("[MODULE_PHASOR] Phasor output shape exceeds the supported range.");
            return Result::ERROR;
        }
    }

    U64 outputPhasorSizeBytes = 0;
    if (!detail::CheckedMultiply(outputPhasorElementCount,
                                 static_cast<U64>(DataTypeSize(DataType::CF32)),
                                 outputPhasorSizeBytes)) {
        JST_ERROR("[MODULE_PHASOR] Phasor output exceeds the supported byte range.");
        return Result::ERROR;
    }

    validatedAntennaPositionTensor = antennaPositions;
    validatedAntennaCalibrationTensor = antennaCalibrations;
    validatedBoresightCoordinateTensor = boresightCoordinates;
    validatedBeamCoordinateTensor = beamCoordinates;
    validatedJulianDateTensor = julianDate;
    validatedDut1Tensor = dut1;
    validatedOutputDelayShape = std::move(outputDelayShape);
    validatedOutputPhasorShape = std::move(outputPhasorShape);
    validatedAntennaCount = antennaCount;
    validatedOutputDelaySizeBytes = outputDelaySizeBytes;
    validatedOutputPhasorSizeBytes = outputPhasorSizeBytes;

    return Result::SUCCESS;
}

Result PhasorImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::STATELESS));

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
    antennaPositionTensor = validatedAntennaPositionTensor;
    antennaCalibrationTensor = validatedAntennaCalibrationTensor;
    boresightCoordinateTensor = validatedBoresightCoordinateTensor;
    beamCoordinateTensor = validatedBeamCoordinateTensor;
    julianDateTensor = validatedJulianDateTensor;
    dut1Tensor = validatedDut1Tensor;

    JST_CHECK(outputDelayTensor.create(
        device(),
        DataType::F64,
        validatedOutputDelayShape
    ));

    // TODO suppoprt frequency axis expansion by some rate
    JST_CHECK(outputPhasorTensor.create(
        device(),
        DataType::CF32,
        validatedOutputPhasorShape
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
