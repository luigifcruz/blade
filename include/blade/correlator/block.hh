#ifndef BLADE_CORRELATOR_BLOCK_HH
#define BLADE_CORRELATOR_BLOCK_HH

#include <string>

#include <jetstream/block.hh>

namespace Jetstream::Blocks {

struct Correlator : public Block::Config {
    U64 integrationRate = 1;
    U64 conjugateAntennaIndex = 1;
    bool useSharedMemory = false;
    std::string calculationMode = "double_precision_fp";
    U64 blockSize = 32;

    JST_BLOCK_TYPE(correlator);
    JST_BLOCK_DOMAIN("BLADE");
    JST_BLOCK_PARAMS(integrationRate, conjugateAntennaIndex, useSharedMemory,
                     calculationMode, blockSize);
    JST_BLOCK_DESCRIPTION(
        "Correlator",
        "Correlates dual-polarization antenna voltages into baseline visibilities.",
        "// TODO: Write description."
    );
};

}  // namespace Jetstream::Blocks

#endif  // BLADE_CORRELATOR_BLOCK_HH
