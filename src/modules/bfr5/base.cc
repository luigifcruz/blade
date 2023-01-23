#define BL_LOG_DOMAIN "M::BFR5"

#include "blade/modules/bfr5/reader.hh"

#include "bfr5.jit.hh"

namespace Blade::Modules::Bfr5 {

Reader::Reader(const Config& config, const Input& input) 
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
    antennaPositions.resize(getDims().numberOfAntennas());
    beamCoordinates.resize(this->bfr5.beam_info.ra_elements);
    beamSourceNames.resize(this->bfr5.beam_info.ra_elements);
     
    // Calculate beam coordinates.
    for (U64 i = 0; i < this->bfr5.beam_info.ra_elements; i++) {
        beamCoordinates[i].RA = this->bfr5.beam_info.ras[i];
        beamCoordinates[i].DEC = this->bfr5.beam_info.decs[i];
        beamSourceNames[i] = std::string(
            (char*) this->bfr5.beam_info.src_names[i].p
        );
    }

    // Calculate antenna positions.
    const U64 antennaPositionsByteSize = getDims().numberOfAntennas() * sizeof(XYZ);
    std::memcpy(antennaPositions.data(), this->bfr5.tel_info.antenna_positions, antennaPositionsByteSize);

    std::string antFrame = std::string(this->bfr5.tel_info.antenna_position_frame);

    if (antFrame != "ecef" && antFrame != "ECEF") {
        if (antFrame == "xyz" || antFrame == "XYZ") {
            calc_position_to_ecef_frame_from_xyz(
                reinterpret_cast<F64*>(antennaPositions.data()),
                antennaPositions.size(),
                this->bfr5.tel_info.latitude,
                this->bfr5.tel_info.longitude,
                this->bfr5.tel_info.altitude);
        }

        if (antFrame == "enu" || antFrame == "ENU") {
            calc_position_to_ecef_frame_from_enu(
                reinterpret_cast<F64*>(antennaPositions.data()),
                antennaPositions.size(),
                this->bfr5.tel_info.latitude,
                this->bfr5.tel_info.longitude,
                this->bfr5.tel_info.altitude);
        }
    }

    BFR5close(&this->bfr5);

    // Print configuration buffers.
    BL_INFO("Input File Path: {}", config.filepath);
    BL_INFO("Data Dimensions [B, A, F, T, P]: {} -> {}", "N/A", getDims());
}

std::vector<CF64> Reader::getAntennaCoefficients(const U64& numberOfFrequencyChannels, const U64& frequencyChannelStartIndex) {
    // transpose from F,P,A to A,F,1,P
    std::vector<CF64> antennaCoefficients;
    const auto coefficientDims = ArrayDimensions({
        .A = this->bfr5.cal_info.cal_all_dims[2],
        .F = numberOfFrequencyChannels == 0 ? this->bfr5.cal_info.cal_all_dims[0] : numberOfFrequencyChannels,
        .T = 1,
        .P = this->bfr5.cal_info.cal_all_dims[1],
    });
    if (frequencyChannelStartIndex + coefficientDims.numberOfFrequencyChannels() > this->bfr5.cal_info.cal_all_dims[0]) {
        BL_FATAL("Requested frequency-channel range [{}, {}) exceeeds dimensions of BFR5 contents ({}).", frequencyChannelStartIndex, frequencyChannelStartIndex + coefficientDims.numberOfFrequencyChannels(), this->bfr5.cal_info.cal_all_dims[0]);
        BL_CHECK_THROW(Result::ASSERTION_ERROR);
    }
    antennaCoefficients.resize(coefficientDims.size());

    const size_t calAntStride = 1;
    const size_t calPolStride = coefficientDims.numberOfAspects() * calAntStride;
    const size_t calChnStride = coefficientDims.numberOfPolarizations() * calPolStride;

    const size_t weightsPolStride = 1;
    const size_t weightsChnStride = coefficientDims.numberOfPolarizations() * weightsPolStride;
    const size_t weightsAntStride = coefficientDims.numberOfFrequencyChannels() * weightsChnStride;

    for (U64 antIdx = 0; antIdx < coefficientDims.numberOfAspects(); antIdx++) {
        for (U64 chnIdx = 0; chnIdx < coefficientDims.numberOfFrequencyChannels(); chnIdx++) {
            for (U64 polIdx = 0; polIdx < coefficientDims.numberOfPolarizations(); polIdx++) {
                const auto inputIdx = (frequencyChannelStartIndex + chnIdx) * calChnStride +
                                        polIdx * calPolStride + 
                                        antIdx * calAntStride;

                const auto outputIdx = antIdx * weightsAntStride +
                                        polIdx * weightsPolStride +
                                        chnIdx * weightsChnStride;

                const auto& coeff = this->bfr5.cal_info.cal_all[inputIdx];
                antennaCoefficients.data()[outputIdx] = {coeff.re, coeff.im};
            }
        }
    }

    return antennaCoefficients;
}

}  // namespace Blade::Modules::Bfr5
