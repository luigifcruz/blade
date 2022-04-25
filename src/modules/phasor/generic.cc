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
        throw Result::ERROR;
    }

    if (config.numberOfAntennas != config.antennaPositions.size()) {
        BL_FATAL("Number of Antennas configuration ({}) mismatches the number of"
                 " antenna positions ({}).", config.numberOfAntennas,
                 config.antennaPositions.size());
        throw Result::ERROR;
    }

    if (config.numberOfAntennas *
        config.numberOfFrequencyChannels * 
        config.numberOfPolarizations
            != config.antennaCalibrations.size()) {
        BL_FATAL("Insufficient number of antenna calibrations ({}). This number"
                 " should be the product of Number of Antennas ({}), Number of"
                 " Frequency Channels ({}), and Number of Polarizations ({}).",
                 config.antennaCalibrations.size(), config.numberOfAntennas,
                 config.numberOfFrequencyChannels, config.numberOfPolarizations);
        throw Result::ERROR;
    }
    
    if (config.referenceAntennaIndex >= config.numberOfAntennas) {
        BL_FATAL("Reference Antenna Index ({}) is larger than the number of"
                 " antennas ({}).", config.referenceAntennaIndex,
                 config.numberOfAntennas);
        throw Result::ERROR;
    }

    BL_INFO("Number of Beams: {}", config.numberOfBeams);
    BL_INFO("Number of Antennas: {}", config.numberOfAntennas);
    BL_INFO("Number of Frequency Channels: {}", config.numberOfFrequencyChannels);
    BL_INFO("Number of Polarizations: {}", config.numberOfPolarizations);
    BL_INFO("RF Frequency (Hz): {}", config.rfFrequencyHz);
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
