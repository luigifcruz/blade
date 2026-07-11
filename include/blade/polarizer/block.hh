#ifndef BLADE_POLARIZER_BLOCK_HH
#define BLADE_POLARIZER_BLOCK_HH

#include <jetstream/block.hh>

namespace Jetstream::Blocks {

struct Polarizer : public Block::Config {
    std::string inputPolarization = "xy";
    std::string outputPolarization = "lr";
    U64 blockSize = 512;

    JST_BLOCK_TYPE(polarizer);
    JST_BLOCK_DOMAIN("BLADE");
    JST_BLOCK_PARAMS(inputPolarization, outputPolarization, blockSize);
    JST_BLOCK_DESCRIPTION(
        "Polarizer",
        "Reorients the polarizations.",
        "# Polarizer\n"
        "The Polarizer block converts a dual linear-polarization complex signal into a "
        "circular basis or extracts a single linear component. Input tensors must be CF32 "
        "shaped as [antennas, channels, samples, polarizations] with two polarizations. "
        "The circular conversion forms the left beam as X plus jY and the right beam as X "
        "minus jY.\n\n"

        "## Arguments\n"
        "- **Input Polarization**: Polarization basis of the input signal, only XY is supported.\n"
        "- **Output Polarization**: Output basis, LR for circular, XY to bypass, or X or Y for a single linear component.\n"
        "- **Block Size**: Number of CUDA threads per block.\n\n"

        "## Useful For\n"
        "- Observing circularly polarized sources with a linear feed.\n"
        "- Splitting a single linear polarization out of a dual-polarization stream.\n"
        "- Preparing voltages for polarization-sensitive detection.\n\n"

        "## Examples\n"
        "- Convert linear to circular polarization:\n"
        "  Config: Input Polarization='xy', Output Polarization='lr'\n"
        "  Input: CF32[20, 192, 8192, 2] -> CF32[20, 192, 8192, 2] in the LR basis.\n\n"

        "## Implementation\n"
        "Input Buffer -> Polarizer Module -> Output Buffer\n"
        "1. Bypasses processing when the input and output bases match, otherwise selects the conversion kernel.\n"
        "2. Multiplies the Y polarization by the ninety degree phasor for the circular basis.\n"
        "3. Writes the converted or extracted polarizations to the output buffer."
    );
};

}  // namespace Jetstream::Blocks

#endif  // BLADE_POLARIZER_BLOCK_HH
