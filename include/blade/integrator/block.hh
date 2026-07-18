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
        "# Integrator\n"
        "The Integrator block sums samples along a tensor axis. The size parameter sums "
        "groups of consecutive indices inside a single buffer, shrinking the axis, while "
        "the rate parameter accumulates successive buffers into the same output before it "
        "is emitted. Input tensors can be F32, CF32, or CI8 of any rank and the output is "
        "always CF32. Size=1 and Rate=1 bypasses integration.\n\n"

        "## Arguments\n"
        "- **Size**: Number of indices summed together within one buffer along the axis.\n"
        "- **Rate**: Number of successive buffers accumulated into one output.\n"
        "- **Axis**: The tensor axis to integrate on, defaulting to the time axis.\n"
        "- **Block Size**: Number of CUDA threads per block. Ignored on CPU.\n\n"

        "## Useful For\n"
        "- Increasing the signal-to-noise ratio of detected spectra.\n"
        "- Reducing the output data rate of a correlator or detector.\n"
        "- Averaging visibilities over longer time spans.\n\n"

        "## Examples\n"
        "- Integrate eight buffers along time:\n"
        "  Config: Size=1, Rate=8, Axis=2\n"
        "  Input: CF32[2, 192, 512, 4] -> CF32[2, 192, 512, 4] emitted every eighth buffer.\n\n"

        "## Implementation\n"
        "Input Buffer -> Integrator Module -> Output Buffer\n"
        "1. Zeroes the output at the start of each accumulation cycle.\n"
        "2. Sums size consecutive groups along the axis inside the buffer.\n"
        "3. Adds the result to the output until rate buffers have been accumulated."
    );
};

}  // namespace Jetstream::Blocks

#endif  // BLADE_INTEGRATOR_BLOCK_HH
