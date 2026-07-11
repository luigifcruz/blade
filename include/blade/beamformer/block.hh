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
        "Forms beams from antenna voltages and phasors.",
        "# Beamformer\n"
        "The Beamformer block forms coherent beams from a phased array by weighting each "
        "antenna voltage with a per-beam phasor and summing over antennas. Input voltages "
        "must be CI8 or CF32 shaped as [antennas, channels, samples, polarizations] and phasors "
        "CF32 shaped as [beams, antennas, channels, 1, polarizations]. An incoherent beam "
        "built from the summed antenna powers can be appended after the coherent beams.\n\n"

        "## Arguments\n"
        "- **Enable Incoherent Beam**: Append an incoherent beam at the last beam index.\n"
        "- **Enable Incoherent Beam Square Root**: Apply a square root to the incoherent beam power.\n"
        "- **Block Size**: Number of CUDA threads per block. The number of time samples must be divisible by this value.\n\n"

        "## Useful For\n"
        "- Steering multiple simultaneous beams from a phased array like the Allen Telescope Array.\n"
        "- Feeding beamformed voltages into detection or spectroscopy pipelines.\n"
        "- Producing an incoherent beam for calibration or comparison against coherent beams.\n\n"

        "## Examples\n"
        "- Form two coherent beams:\n"
        "  Config: defaults\n"
        "  Input: CI8 or CF32[20, 192, 8192, 2] + Phasors CF32[2, 20, 192, 1, 2] -> CF32[2, 192, 8192, 2].\n\n"

        "## Implementation\n"
        "Input Buffer + Phasors -> Beamformer Module -> Output Buffer\n"
        "1. Caches the beam phasors in shared memory and the antenna samples in registers.\n"
        "2. Multiplies each antenna voltage by its beam phasor and accumulates over antennas.\n"
        "3. Optionally detects the per-antenna power and accumulates it into the incoherent beam."
    );
};

}  // namespace Jetstream::Blocks

#endif  // BLADE_BEAMFORMER_BLOCK_HH
