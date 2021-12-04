#include "advanced.hh"
#include "blade/modules/beamformer/ata.hh"

using namespace Blade;

int main() {
    Logger guard{};

    BL_INFO("Testing beamformer with the ATA kernel.");

    Test<CF32, CF32> mod({
        .dims = {
            .NBEAMS = 16,
            .NANTS  = 20,
            .NCHANS = 192,
            .NTIME  = 8192,
            .NPOLS  = 2,
        },
        .blockSize = 512,
    });

    Memory::HostVector<CF32> input(mod.getInputSize());
    Memory::HostVector<CF32> phasors(mod.getPhasorsSize());
    Memory::HostVector<CF32> output(mod.getOutputSize());

    for (int i = 0; i < 24; i++) {
        if (mod.run(input, phasors, output) != Result::SUCCESS) {
            BL_WARN("Fault was encountered. Test is exiting...");
            return 1;
        }
    }

    BL_INFO("Test succeeded.");

    return 0;
}
