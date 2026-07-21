#ifndef BLADE_DETECTOR_BLOCK_HH
#define BLADE_DETECTOR_BLOCK_HH

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
        "Integrates detected power products.",
        "# Detector\n"
        "The Detector block detects power products from dual-polarization complex voltages "
        "and integrates them over time. Input tensors must be CF32 shaped as [antennas, "
        "channels, samples, polarizations] with two polarizations. With four output "
        "polarizations it produces XX, YY, and the real and imaginary parts of XY, and "
        "with one it produces the total power XX plus YY.\n\n"

        "## Arguments\n"
        "- **Integration Rate**: Number of time samples summed into each output sample.\n"
        "- **Number of Output Polarizations**: Detected products per sample, 1 or 4.\n"
        "- **Block Size**: Number of CUDA threads per block. Ignored on CPU.\n\n"

        "## Useful For\n"
        "- Converting beamformed voltages into power spectra.\n"
        "- Reducing the data rate ahead of a spectrometer sink.\n"
        "- Producing full polarization products for Stokes analysis.\n\n"

        "## Examples\n"
        "- Detect total power with integration:\n"
        "  Config: Integration Rate=16, Number of Output Polarizations=1\n"
        "  Input: CF32[2, 192, 8192, 2] -> F32[2, 192, 512, 1].\n\n"

        "## Implementation\n"
        "Input Buffer -> Detector Module -> Output Buffer\n"
        "1. Selects the one or four polarization CPU loop or CUDA kernel.\n"
        "2. Computes the power products of each dual-polarization sample.\n"
        "3. Accumulates every integration window into one output sample."
    );
};

}  // namespace Jetstream::Blocks

#endif  // BLADE_DETECTOR_BLOCK_HH
