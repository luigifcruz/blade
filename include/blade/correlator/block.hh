#ifndef BLADE_CORRELATOR_BLOCK_HH
#define BLADE_CORRELATOR_BLOCK_HH

#include <string>

#include <jetstream/block.hh>

namespace Jetstream::Blocks {

struct Correlator : public Block::Config {
    U64 integrationRate = 1;
    U64 conjugateAntennaIndex = 1;
    std::string calculationMode = "double_precision_fp";

    JST_BLOCK_TYPE(correlator);
    JST_BLOCK_DOMAIN("BLADE");
    JST_BLOCK_NODE_SIZE(M);
    JST_BLOCK_PARAMS(integrationRate, conjugateAntennaIndex, calculationMode);
    JST_BLOCK_DESCRIPTION(
        "Correlator",
        "Correlates voltages into baseline visibilities.",
        "# Correlator\n"
        "The Correlator block cross-correlates dual-polarization antenna voltages into "
        "baseline visibilities. Input tensors must be CI8 or CF32 shaped as [antennas, "
        "channels, samples, polarizations] and every antenna pair, including the "
        "autocorrelations, produces the four polarization products XX, XY, YX, and YY "
        "summed over the time axis. Successive input buffers can be accumulated into a "
        "single output with the integration rate.\n\n"

        "## Arguments\n"
        "- **Integration Rate**: Number of input buffers accumulated into each output buffer.\n"
        "- **Conjugate Antenna Index**: Which antenna of the pair is conjugated, 0 for A and 1 for B.\n"
        "- **Calculation Mode**: Intermediate calculation precision.\n\n"

        "## Useful For\n"
        "- Producing visibilities for radio interferometric imaging.\n"
        "- Feeding UVH5 writers with integrated baseline products.\n"
        "- Monitoring the array health through autocorrelations.\n\n"

        "## Examples\n"
        "- Correlate twenty antennas:\n"
        "  Config: Integration Rate=8\n"
        "  Input: CF32[20, 192, 8192, 2] -> CF32[210, 192, 1, 4] emitted every eighth buffer.\n\n"

        "## Implementation\n"
        "Input Buffer -> Correlator Module -> Output Buffer\n"
        "1. Zeroes the output visibilities at the start of each integration window.\n"
        "2. Stages every antenna's voltages for one channel and time slice into shared memory.\n"
        "3. Reduces a 2x2 tile of the baseline matrix per thread, holding the products in registers.\n"
        "4. Accumulates each buffer into the output with exactly one addition per visibility, "
        "so the result is bit-reproducible across runs, and emits the buffer when the window closes."
    );
};

}  // namespace Jetstream::Blocks

#endif  // BLADE_CORRELATOR_BLOCK_HH
