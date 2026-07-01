#ifndef BLADE_INTEGRATOR_BLOCK_HH
#define BLADE_INTEGRATOR_BLOCK_HH

#include <jetstream/block.hh>

namespace Jetstream::Blocks {

struct Integrator : public Block::Config {
    U64 size = 1;  // Number of indices to integrate within one block.
    U64 rate = 1;  // Number of blocks to integrate together.
    U64 axis = 2;  // The block axis to integrate on, defaulting to T
    U64 blockSize = 512;

    JST_BLOCK_TYPE(integrator);
    JST_BLOCK_DOMAIN("BLADE");
    JST_BLOCK_PARAMS(size, rate, axis, blockSize);
    JST_BLOCK_DESCRIPTION(
        "Integrator",
        "Integrates (sums) samples along an axis.",
        "// TODO: Write description."
    );
};

}  // namespace Jetstream::Blocks

#endif  // BLADE_INTEGRATOR_BLOCK_HH
