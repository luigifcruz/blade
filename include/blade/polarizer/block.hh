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
        "// TODO: Write description."
    );
};

}  // namespace Jetstream::Blocks

#endif  // BLADE_POLARIZER_BLOCK_HH
