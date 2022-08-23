#define BL_LOG_DOMAIN "M::PHASOR"

#include <algorithm>

#include "blade/modules/phasor/generic.hh"

#include "phasor.jit.hh"

namespace Blade::Modules::Phasor {

template<typename OT>
Generic<OT>::Generic(const Config& config, const Input& input)
        : Module(config.blockSize, phasor_kernel),
          config(config),
          input(input) {
    if (config.numberOfBeams != config.beamCoordinates.size()) {
        BL_FATAL("Number of Beams configuration ({}) mismatches the number of"
                 " beams coordinates ({}).", config.numberOfBeams,
                 config.beamCoordinates.size());
        BL_CHECK_THROW(Result::ERROR);
    }

    if (config.numberOfAntennas != config.antennaPositions.size()) {
        BL_FATAL("Number of Antennas configuration ({}) mismatches the number of"
                 " antenna positions ({}).", config.numberOfAntennas,
                 config.antennaPositions.size());
        BL_CHECK_THROW(Result::ERROR);
    }
    
    if (config.referenceAntennaIndex >= config.numberOfAntennas) {
        BL_FATAL("Reference Antenna Index ({}) is larger than the number of"
                 " antennas ({}).", config.referenceAntennaIndex,
                 config.numberOfAntennas);
        BL_CHECK_THROW(Result::ERROR);
    }

    const F64& max_value = (65500.0 / (config.numberOfAntennas * 127.0));
    const F64& min_value = max_value * -1.0;

    F64 max_cal = 0.0, min_cal = 0.0;
    for (const auto& calibration : config.antennaCalibrations) {
        if (calibration.real() > max_cal) {
            max_cal = calibration.real();
        }

        if (calibration.imag() > max_cal) {
            max_cal = calibration.imag();
        }

        if (calibration.real() < min_cal) {
            min_cal = calibration.real();
        }

        if (calibration.imag() < min_cal) {
            min_cal = calibration.imag();
        }
    }

    if ((max_value < max_cal) || ((min_value > min_cal))) {
        BL_WARN("Overflow Possible! At least one calibration value is smaller" 
                " or larger ({:.2f}, {:.2f}) than what CF16 can hold ({:.2f}, {:.2f})"
                " with current configuration parameters.",
                min_cal, max_cal, min_value, max_value); 
    }

    BL_INFO("Output Type: {}", TypeInfo<OT>::name);
    BL_INFO("Number of Beams: {}", config.numberOfBeams);
    BL_INFO("Number of Antennas: {}", config.numberOfAntennas);
    BL_INFO("Number of Frequency Channels: {}", config.numberOfFrequencyChannels);
    BL_INFO("Number of Polarizations: {}", config.numberOfPolarizations);
    BL_INFO("Observation Frequency (Hz): {}", config.observationFrequencyHz);
    BL_INFO("Channel Bandwidth (Hz): {}", config.channelBandwidthHz);
    BL_INFO("Total Bandwidth (Hz): {}", config.totalBandwidthHz);
    BL_INFO("Frequency Start Index: {}", config.frequencyStartIndex);
    BL_INFO("Reference Antenna Index: {}", config.referenceAntennaIndex);
    BL_INFO("Array Reference Position (LON, LAT, ALT): ({}, {}, {})",
        config.arrayReferencePosition.LON, config.arrayReferencePosition.LAT,
        config.arrayReferencePosition.ALT);
    BL_INFO("Boresight Coordinate (RA, DEC): ({}, {})",
        config.boresightCoordinate.RA, config.boresightCoordinate.DEC);

    BL_INFO("ECEF Antenna Positions (X, Y, Z):");
    for (U64 i = 0; i < config.antennaPositions.size(); i++) {
        BL_INFO("    {}: ({}, {}, {})", i, config.antennaPositions[i].X, 
            config.antennaPositions[i].Y, config.antennaPositions[i].Z);
    }

    BL_INFO("Beam Coordinates (RA, DEC):");
    for (U64 i = 0; i < config.beamCoordinates.size(); i++) {
        BL_INFO("    {}: ({}, {})", i, config.beamCoordinates[i].RA, 
            config.beamCoordinates[i].DEC);
    }
}

template class BLADE_API Generic<CF32>;
template class BLADE_API Generic<CF64>;

}  // namespace Blade::Modules::Phasor
