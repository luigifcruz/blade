#include <span>

#include "bl-beamformer/base.hh"

using namespace BL;

Result Run(const Beamformer::Config & config);

int main() {
    Logger::Init();

    BL_INFO("Testing beamformer with the MeetKAT kernel.");

    Beamformer::Config config = {
        .NBEAMS = 64,
        .NANTS  = 64,
        .NCHANS = 1,
        .NTIME  = 4194304,
        .NPOLS  = 2,
        .TBLOCK = 256,
        .kernel = Beamformer::Kernel::MEERKAT,
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
