#include <span>

#include "bl-beamformer/base.hh"

using namespace BL;

Result Run(const Beamformer::Config & config);

int main() {
    Logger::Init();

    BL_INFO("Testing beamformer with the ATA kernel.");

    Beamformer::Config config = {
        .NBEAMS = 16,
        .NANTS  = 20,
        .NCHANS = 384,
        .NTIME  = 8750,
        .NPOLS  = 2,
        .TBLOCK = 350,
        .kernel = Beamformer::Kernel::ATA,
    };

    if (Run(config) != Result::SUCCESS) {
        BL_FATAL("Test failed.");
        Logger::Shutdown();
        return 1;
    }

    BL_INFO("Test succeeded.");
    Logger::Shutdown();

    return 0;
}
