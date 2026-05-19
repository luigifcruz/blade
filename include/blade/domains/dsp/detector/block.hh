#ifndef BLADE_DOMAINS_DSP_DETECTOR_BLOCK_HH
#define BLADE_DOMAINS_DSP_DETECTOR_BLOCK_HH

#include <jetstream/block.hh>

namespace Jetstream::Blocks {

struct Detector : public Block::Config {
    U64 integrationRate = 1;
    U64 numberOfOutputPolarizations = 4;
    U64 blockSize = 512;

    JST_BLOCK_TYPE(detector);
    JST_BLOCK_DOMAIN("BLADE");
    JST_BLOCK_PARAMS(integrationRate, numberOfOutputPolarizations, blockSize);
    JST_BLOCK_DESCRIPTION(
        "Detector",
        "Integrates detected power products from dual-polarization complex input.",
        "// TODO: Write description."
    );
};

}  // namespace Jetstream::Blocks

#endif  // BLADE_DOMAINS_DSP_DETECTOR_BLOCK_HH
