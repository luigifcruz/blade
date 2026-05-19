#ifndef BLADE_BEAMFORMER_BLOCK_HH
#define BLADE_BEAMFORMER_BLOCK_HH

#include <jetstream/block.hh>

namespace Jetstream::Blocks {

struct Beamformer : public Block::Config {
    bool enableIncoherentBeam = false;
    bool enableIncoherentBeamSqrt = false;
    U64 blockSize = 512;

    JST_BLOCK_TYPE(beamformer);
    JST_BLOCK_DOMAIN("BLADE");
    JST_BLOCK_PARAMS(enableIncoherentBeam, enableIncoherentBeamSqrt, blockSize);
    JST_BLOCK_DESCRIPTION(
        "Beamformer",
        "Forms coherent and optional incoherent beams from antenna voltages and phasors.",
        "// TODO: Write description."
    );
};

}  // namespace Jetstream::Blocks

#endif  // BLADE_BEAMFORMER_BLOCK_HH
