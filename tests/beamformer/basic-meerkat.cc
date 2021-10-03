#include "blade/kernels/beamformer.hh"

using namespace Blade;

Result Run(const Kernel::Beamformer::Config & config);

int main() {
    Logger guard{};

    BL_INFO("Testing beamformer with the MeetKAT kernel.");

    Kernel::Beamformer::Config config = {
        .NBEAMS = 64,
        .NANTS  = 64,
        .NCHANS = 1,
        .NTIME  = 4194304,
        .NPOLS  = 2,
        .TBLOCK = 256,
        .recipe = Kernel::Beamformer::Recipe::MEERKAT,
    };

    if (Run(config) != Result::SUCCESS) {
        BL_FATAL("Test failed.");
        return 1;
    }

    BL_INFO("Test succeeded.");

    return 0;
}
