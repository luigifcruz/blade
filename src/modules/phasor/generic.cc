#define BL_LOG_DOMAIN "M::PHASOR"

#include <algorithm>

#include "blade/modules/phasor/generic.hh"

#include "phasor.jit.hh"

namespace Blade::Modules::Phasor {

template<typename OT>
Generic<OT>::Generic(const Config& config, const Input& input)
        : Module(phasor_program),
          config(config),
          input(input) {
    // Check configuration values.
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

    // Check if coefficient values are within bounds.
    const F64& max_value = (65500.0 / (config.numberOfAntennas * 127.0));
    const F64& min_value = max_value * -1.0;

    F64 max_coeff = 0.0, min_coeff = 0.0;
    for (const auto& coefficient : config.antennaCoefficients) {
        if (coefficient.real() > max_coeff) {
            max_coeff = coefficient.real();
        }

        if (coefficient.imag() > max_coeff) {
            max_coeff = coefficient.imag();
        }

        if (coefficient.real() < min_coeff) {
            min_coeff = coefficient.real();
        }

        if (coefficient.imag() < min_coeff) {
            min_coeff = coefficient.imag();
        }
    }

    if ((max_value < max_coeff) || ((min_value > min_coeff))) {
        BL_WARN("Overflow Possible! At least one coefficient value is smaller" 
                " or larger ({:.2f}, {:.2f}) than what CF16 can hold ({:.2f}, {:.2f})"
                " with current configuration parameters.",
                min_coeff, max_coeff, min_value, max_value); 
    }

    // Print generic configuration values.
    BL_INFO("Type: {} -> {}", "N/A", TypeInfo<OT>::name);
    BL_INFO("Delays Negated: {}", config.negateDelays);
    BL_INFO("Bottom Frequency (Hz): {}", config.bottomFrequencyHz);
    BL_INFO("Channel Bandwidth (Hz): {}", config.channelBandwidthHz);
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
