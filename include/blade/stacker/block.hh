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
        "# Stacker\n"
        "The Stacker block tiles successive input buffers along a tensor axis, assembling "
        "an output whose axis is ratio times larger than the input. Each compute cycle "
        "writes the current buffer into its slot and the output is emitted once every "
        "ratio buffers. Input tensors can be F32, CF32, or CI8 and can have any rank. "
        "Ratio=1 bypasses stacking.\n\n"

        "## Arguments\n"
        "- **Axis**: The tensor axis to stack along.\n"
        "- **Ratio**: Number of buffers tiled into one output, must be greater than zero.\n"
        "- **Copy Size Threshold**: Chunk width in elements below which a kernel is used instead of a strided memcopy.\n"
        "- **Block Size**: Number of CUDA threads per block.\n\n"

        "## Useful For\n"
        "- Batching short buffers into larger blocks for downstream processing.\n"
        "- Building waterfall displays from successive spectra.\n"
        "- Matching the buffer geometry expected by writers and correlators.\n\n"

        "## Examples\n"
        "- Stack eight buffers along time:\n"
        "  Config: Axis=2, Ratio=8\n"
        "  Input: CF32[2, 192, 1024, 2] -> CF32[2, 192, 8192, 2] emitted every eighth buffer.\n\n"

        "## Implementation\n"
        "Input Buffer -> Stacker Module -> Output Buffer\n"
        "1. Zeroes the output when a new stacking cycle begins.\n"
        "2. Copies the buffer into its slot with a kernel for narrow chunks or a strided memcopy for wide ones.\n"
        "3. Emits the assembled output after ratio buffers have been placed."
    );
};

}  // namespace Jetstream::Blocks

#endif  // BLADE_STACKER_BLOCK_HH
