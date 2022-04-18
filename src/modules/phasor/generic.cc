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

    BL_DEBUG("Number of Beams: {}", config.numberOfBeams);
    BL_DEBUG("Number of Antennas: {}", config.numberOfAntennas);
    BL_DEBUG("Number of Frequency Channels: {}", config.numberOfFrequencyChannels);
    BL_DEBUG("Number of Polarizations: {}", config.numberOfPolarizations);
    BL_DEBUG("RF Frequency (Hz): {}", config.rfFrequencyHz);
    BL_DEBUG("Channel Bandwidth (Hz): {}", config.channelBandwidthHz);
    BL_DEBUG("Total Bandwidth (Hz): {}", config.totalBandwidthHz);
    BL_DEBUG("Frequency Start Index: {}", config.frequencyStartIndex);
    BL_DEBUG("Reference Antenna Index: {}", config.referenceAntennaIndex);
    BL_DEBUG("Array Reference Position (LON, LAT, ALT): ({}, {}, {})",
        config.arrayReferencePosition.LON, config.arrayReferencePosition.LAT,
        config.arrayReferencePosition.ALT);
    BL_DEBUG("Boresight Coordinate (RA, DEC): ({}, {})",
        config.boresightCoordinate.RA, config.boresightCoordinate.DEC);

    BL_DEBUG("ECEF Antenna Positions (X, Y, Z):");
    for (U64 i = 0; i < config.antennaPositions.size(); i++) {
        BL_DEBUG("    {}: ({}, {}, {})", i, config.antennaPositions[i].X, 
            config.antennaPositions[i].Y, config.antennaPositions[i].Z);
    }

    BL_DEBUG("Beam Coordinates (RA, DEC):");
    for (U64 i = 0; i < config.beamCoordinates.size(); i++) {
        BL_DEBUG("    {}: ({}, {})", i, config.beamCoordinates[i].RA, 
            config.beamCoordinates[i].DEC);
    }
}

template class BLADE_API Generic<CF32>;
template class BLADE_API Generic<CF64>;

}  // namespace Blade::Modules::Phasor
