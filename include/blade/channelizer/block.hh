#ifndef BLADE_CHANNELIZER_BLOCK_HH
#define BLADE_CHANNELIZER_BLOCK_HH

#include <jetstream/block.hh>

namespace Jetstream::Blocks {

struct Channelizer : public Block::Config {
    JST_BLOCK_TYPE(channelizer);
    JST_BLOCK_DOMAIN("BLADE");
    JST_BLOCK_PARAMS();
    JST_BLOCK_DESCRIPTION(
        "Channelizer",
        "Splits frequency into voltages channels.",
        "# Channelizer\n"
        "The Channelizer applies a centered FFT to the time axis of a contiguous CF32 "
        "voltage tensor shaped [aspects, coarse channels, samples, polarizations]. It "
        "then merges each coarse channel with its fine-frequency bins while preserving "
        "BLADE's four-dimensional tensor convention. This is a plain FFT channelizer, "
        "not a polyphase filter bank.\n\n"

        "## Useful For\n"
        "- Splitting coarse telescope channels into centered fine-frequency bins.\n"
        "- Preparing voltage spectra for beamforming, detection, or correlation.\n\n"

        "## Examples\n"
        "- Full-buffer channelization:\n"
        "  Input: CF32[8, 128, 1024, 2] -> Output: CF32[8, 131072, 1, 2]\n\n"

        "## Implementation\n"
        "Input Buffer -> Shifted FFT -> Reshape -> Output Buffer\n"
        "1. Alternating signs center the even-length FFT output.\n"
        "2. An unnormalized forward FFT transforms the time axis.\n"
        "3. A zero-copy reshape merges coarse and fine-frequency dimensions."
    );
};

}  // namespace Jetstream::Blocks

#endif  // BLADE_CHANNELIZER_BLOCK_HH
