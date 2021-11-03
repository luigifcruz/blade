#include "advanced.hh"
#include "blade/beamformer/ata.hh"
#include "blade/beamformer/ata_test.hh"

using namespace Blade;

int main() {
    Logger guard{};
    Manager manager{};

    BL_INFO("Testing beamformer with the ATA kernel.");

    Module<Beamformer::ATA> mod({
        .dims = {
            .NBEAMS = 16,
            .NANTS  = 20,
            .NCHANS = 192,
            .NTIME  = 8192,
            .NPOLS  = 2,
        },
        .blockSize = 512,
    });

    manager.save(mod).report();

    for (int i = 0; i < 24; i++) {
        if (mod.run() != Result::SUCCESS) {
            BL_WARN("Fault was encountered. Test is exiting...");
            return 1;
        }
    }

    BL_INFO("Test succeeded.");

    return 0;
}
