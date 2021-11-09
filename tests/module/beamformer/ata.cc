#include "advanced.hh"
#include "blade/modules/beamformer/ata.hh"
#include "blade/modules/beamformer/ata_test.hh"

using namespace Blade;

int main() {
    Logger guard{};
    Manager manager{};

    BL_INFO("Testing beamformer with the ATA kernel.");

    Module<Modules::Beamformer::ATA> mod({
        .dims = {
            .NBEAMS = 16,
            .NANTS  = 20,
            .NCHANS = 192,
            .NTIME  = 8192,
            .NPOLS  = 2,
        },
        .blockSize = 512,
    });

    manager.save(mod.getResources()).report();

    for (int i = 0; i < 24; i++) {
        if (mod.run() != Result::SUCCESS) {
            BL_WARN("Fault was encountered. Test is exiting...");
            return 1;
        }
    }

    BL_INFO("Test succeeded.");

    return 0;
}
