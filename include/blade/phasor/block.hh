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
        "// TODO: Write description."
    );
};

}  // namespace Jetstream::Blocks

#endif  // BLADE_PHASOR_BLOCK_HH
