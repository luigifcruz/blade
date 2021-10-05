#include "blade/beamformer/meerkat.hh"

using namespace Blade;

Result Run(Beamformer::Generic &);

int main() {
    Logger guard{};

    BL_INFO("Testing beamformer with the MeetKAT kernel.");

    Beamformer::MeerKAT beam({
        .NBEAMS = 64,
        .NANTS  = 64,
        .NCHANS = 1,
        .NTIME  = 4194304,
        .NPOLS  = 2,
        .TBLOCK = 256,
    });

    if (Run(beam) != Result::SUCCESS) {
        BL_FATAL("Test failed.");
        return 1;
    }

    BL_INFO("Test succeeded.");

    return 0;
}
