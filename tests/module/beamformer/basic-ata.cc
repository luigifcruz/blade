#include "basic.hh"
#include "blade/beamformer/ata.hh"

using namespace Blade;

int main() {
    Logger guard{};
    Manager manager{};

    BL_INFO("Testing beamformer with the ATA kernel.");

    Module<Beamformer::ATA> mod({
        {
            .NBEAMS = 16,
            .NANTS  = 20,
            .NCHANS = 384,
            .NTIME  = 8750,
            .NPOLS  = 2,
        }, {
            .blockSize = 350,
        },
    });

    manager.save(mod).report();

    for (int i = 0; i < 150; i++) {
        if (mod.process(true) != Result::SUCCESS) {
            BL_WARN("Fault was encountered. Test is exiting...");
            return 1;
        }
    }

    BL_INFO("Test succeeded.");

    return 0;
}
