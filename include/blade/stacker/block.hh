#ifndef BLADE_STACKER_BLOCK_HH
#define BLADE_STACKER_BLOCK_HH

#include <jetstream/block.hh>

namespace Jetstream::Blocks {

struct Stacker : public Block::Config {
    U64 axis = 0;
    U64 ratio = 1;
    U64 copySizeThreshold = 512;
    U64 blockSize = 512;

    JST_BLOCK_TYPE(stacker);
    JST_BLOCK_DOMAIN("BLADE");
    JST_BLOCK_PARAMS(axis, ratio, copySizeThreshold, blockSize);
    JST_BLOCK_DESCRIPTION(
        "Stacker",
        "Stacks the input along a specified axis.",
        "// TODO: Write description."
    );
};

}  // namespace Jetstream::Blocks

#endif  // BLADE_STACKER_BLOCK_HH
