#ifndef BLADE_PHASOR_BLOCK_HH
#define BLADE_PHASOR_BLOCK_HH

#include <jetstream/block.hh>

namespace Jetstream::Blocks {

struct Phasor : public Block::Config {
    F64 observationFrequencyHz = 1.4e9;
    F64 channelBandwidthHz = 1e6;
    F64 totalBandwidthHz = 1e6;
    U64 frequencyStartIndex = 0;

    U64 referenceAntennaIndex = 0;
    F64 arrayReferenceLongitude = -52.874986919708704*M_PI/180;
    F64 arrayReferenceLatitude = -40.00980599999271*M_PI/180;
    F64 arrayReferenceAltitude = 818.6837417125288;

    JST_BLOCK_TYPE(phasor);
    JST_BLOCK_DOMAIN("BLADE");
    JST_BLOCK_PARAMS(
        observationFrequencyHz,
        channelBandwidthHz,
        totalBandwidthHz,
        frequencyStartIndex,
        referenceAntennaIndex,
        arrayReferenceLongitude,
        arrayReferenceLatitude,
        arrayReferenceAltitude
    );
    JST_BLOCK_DESCRIPTION(
        "Phasor",
        "Calculates the phasors for beamforming.",
        "# Phasor\n"
        "The Phasor block computes the geometric delays and complex phasor weights that "
        "steer a beamformer. It takes the ECEF antenna positions, per-antenna calibrations, "
        "the boresight coordinates, and the per-beam right ascension and declination, and "
        "produces one delay per beam and antenna plus one calibrated phasor per beam, "
        "antenna, channel, and polarization. All angles are in radians and all frequencies "
        "are in Hertz.\n\n"

        "## Arguments\n"
        "- **Observation Frequency**: Center frequency of the observation in hertz.\n"
        "- **Channel Bandwidth**: Bandwidth of a single frequency channel in hertz.\n"
        "- **Total Bandwidth**: Total bandwidth of the observation in hertz.\n"
        "- **Frequency Start Index**: Zero-based index of the first frequency channel being processed.\n"
        "- **Reference Antenna Index**: Antenna used as the delay reference.\n"
        "- **Array Reference Longitude**: Longitude of the array reference position in radians.\n"
        "- **Array Reference Latitude**: Latitude of the array reference position in radians.\n"
        "- **Array Reference Altitude**: Altitude of the array reference position in meters.\n\n"

        "## Useful For\n"
        "- Driving the Beamformer block with delay-based steering weights.\n"
        "- Applying per-antenna bandpass calibrations while beamforming.\n"
        "- Tracking sources as the delays evolve with the Julian date input.\n\n"

        "## Examples\n"
        "- Compute phasors for two beams:\n"
        "  Config: defaults\n"
        "  Input: Positions F64[20, 3] + Calibrations CF64[20, 192, 2] + Coordinates -> Phasors CF32[2, 20, 192, 2].\n\n"

        "## Implementation\n"
        "Antenna Positions + Coordinates -> Phasor Module -> Delays + Phasors\n"
        "1. Converts the ECEF antenna positions to the array-centered XYZ frame.\n"
        "2. Rotates the positions towards the boresight and each beam to compute relative delays with radiointerferometryc99.\n"
        "3. Synthesizes the complex phasors from the delays and multiplies them by the antenna calibrations."
    );
};

}  // namespace Jetstream::Blocks

#endif  // BLADE_PHASOR_BLOCK_HH
