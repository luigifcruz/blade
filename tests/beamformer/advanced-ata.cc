#include "blade/instruments/beamformer/test/ata.hh"
#include "blade/kernels/beamformer.hh"

using namespace Blade;

Result Run(const Kernel::Beamformer::Config &, Instrument::Beamformer::Test::Generic &);

int main() {
    Logger guard{};

    BL_INFO("Testing beamformer with the ATA kernel.");

    Instrument::Beamformer::Test::ATA test;

    Kernel::Beamformer::Config config = {
        .NBEAMS = 16,
        .NANTS  = 20,
        .NCHANS = 384,
        .NTIME  = 8750,
        .NPOLS  = 2,
        .TBLOCK = 350,
        .recipe = Kernel::Beamformer::Recipe::ATA,
    };

    if (Run(config, test) != Result::SUCCESS) {
        BL_FATAL("Test failed.");
        return 1;
    }

    BL_INFO("Test succeeded.");

    return 0;
}
