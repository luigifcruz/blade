#define BL_LOG_DOMAIN "M::BFR5"

#include "blade/modules/bfr5/reader.hh"

#include "bfr5.jit.hh"

namespace Blade::Modules::Bfr5 {

Reader::Reader(const Config& config,
               const Input& input,
               const cudaStream_t& stream) 
        : Module(bfr5_program),
          config(config),
          input(input) {
    // Check configuration values.
    if (!std::filesystem::exists(config.filepath)) {
        BL_FATAL("Input file ({}) doesn't not exist.", config.filepath);
        BL_CHECK_THROW(Result::ASSERTION_ERROR);
    }

    // Open header file.
    BFR5open(config.filepath.c_str(), &this->bfr5);
    BFR5read_all(&this->bfr5);

    // Resize data holders.
    beamCoordinates.resize(this->bfr5.beam_info.ra_elements);
    antennaPositions.resize(getTotalShape().numberOfAspects());
    antennaCalibrations = ArrayTensor<Device::CPU, CF64>(getAntennaCalibrationsShape());
     
    // Calculate beam coordinates.
    for (U64 i = 0; i < this->bfr5.beam_info.ra_elements; i++) {
        beamCoordinates[i].RA = this->bfr5.beam_info.ras[i];
        beamCoordinates[i].DEC = this->bfr5.beam_info.decs[i];
    }

    // Calculate antenna positions.
    const U64 antennaPositionsByteSize = getTotalShape().numberOfAspects() * sizeof(XYZ);
    std::memcpy(antennaPositions.data(), this->bfr5.tel_info.antenna_positions, antennaPositionsByteSize);

    std::string antFrame = std::string(this->bfr5.tel_info.antenna_position_frame);

    if (antFrame != "xyz" && antFrame != "XYZ") {
        if (antFrame == "ecef" || antFrame == "ECEF") {
            calc_position_to_xyz_frame_from_ecef(
                reinterpret_cast<F64*>(antennaPositions.data()),
                antennaPositions.size(),
                this->bfr5.tel_info.latitude,
                this->bfr5.tel_info.longitude,
                this->bfr5.tel_info.altitude);
        }

        if (antFrame == "enu" || antFrame == "ENU") {
            calc_position_to_xyz_frame_from_enu(
                reinterpret_cast<F64*>(antennaPositions.data()),
                antennaPositions.size(),
                this->bfr5.tel_info.latitude,
                this->bfr5.tel_info.longitude,
                this->bfr5.tel_info.altitude);
        }
    }

    BFR5close(&this->bfr5);

    // Calculate antenna calibrations.
    const size_t calAntStride = 1;
    const size_t calPolStride = getAntennaCalibrationsShape().numberOfAspects() * calAntStride;
    const size_t calChnStride = getAntennaCalibrationsShape().numberOfPolarizations() * calPolStride;

    const size_t weightsPolStride = 1;
    const size_t weightsChnStride = getAntennaCalibrationsShape().numberOfPolarizations() * weightsPolStride;
    const size_t weightsAntStride = getTotalShape().numberOfFrequencyChannels() * weightsChnStride;

    for (U64 antIdx = 0; antIdx < getAntennaCalibrationsShape().numberOfAspects(); antIdx++) {
        for (U64 chnIdx = 0; chnIdx < getTotalShape().numberOfFrequencyChannels(); chnIdx++) {
            for (U64 polIdx = 0; polIdx < getAntennaCalibrationsShape().numberOfPolarizations(); polIdx++) {
                for (U64 fchIdx = 0; fchIdx < config.channelizerRate; fchIdx++) {
                    const auto inputIdx = chnIdx * calChnStride +
                                          polIdx * calPolStride + 
                                          antIdx * calAntStride;
                    const auto frqIdx = chnIdx * config.channelizerRate + fchIdx;
                    const auto outputIdx = antIdx * weightsAntStride +
                                           polIdx * weightsPolStride +
                                           frqIdx * weightsChnStride;
                    const auto& coeff = this->bfr5.cal_info.cal_all[inputIdx];
                    antennaCalibrations[outputIdx] = {coeff.re, coeff.im};
                }
            }
        }
    }

    // Print configuration buffers.
    BL_INFO("Input File Path: {}", config.filepath);
    BL_INFO("Calibrations Shape: {} -> {}", "N/A", getAntennaCalibrationsShape().str());
    BL_INFO("Data Shape: {} -> {}", "N/A", getTotalShape().str());
    BL_INFO("Channelizer Rate: {}", config.channelizerRate);
}

}  // namespace Blade::Modules::Bfr5
