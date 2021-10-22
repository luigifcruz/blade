#include "blade/beamformer/ata.hh"

using namespace Blade;

Result Run(Beamformer::Generic &);

int main() {
    Logger guard{};

    BL_INFO("Testing beamformer with the ATA kernel.");

    Beamformer::ATA beam({
        {
            .NBEAMS = 16,
            .NANTS  = 20,
            .NCHANS = 384,
            .NTIME  = 8750,
            .NPOLS  = 2,
        },
        350,
    });

    if (Run(beam) != Result::SUCCESS) {
        BL_WARN("Fault was encountered. Test is exiting...");
        return 1;
    }

    BL_INFO("Test succeeded.");

    return 0;
}
