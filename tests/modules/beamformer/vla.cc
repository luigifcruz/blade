#include "pipeline_vla.hh"
#include "blade/modules/beamformer/vla.hh"

using namespace Blade;

int main() {
    BL_INFO("Testing beamformer with the VLA kernel.");

    Test<CF32, CF32> mod({
        .numberOfBeams = 16,
        .numberOfAntennas = 20,
        .numberOfFrequencyChannels = 192,
        .numberOfTimeSamples = 8192,
        .numberOfPolarizations = 2,
        .blockSize = 512,
    });

    Vector<Device::CPU, CF32> input(mod.getInputSize());
    Vector<Device::CPU, CF32> phasors(mod.getPhasorsSize());
    Vector<Device::CPU, CF32> output(mod.getOutputSize());

    for (int i = 0; i < 24; i++) {
        if (mod.run(input, phasors, output, true) != Result::SUCCESS) {
            BL_WARN("Fault was encountered. Test is exiting...");
            return 1;
        }
    }

    BL_INFO("Test succeeded.");

    return 0;
}
