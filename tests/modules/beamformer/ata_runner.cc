#include "pipeline.hh"
#include "blade/modules/beamformer/ata.hh"
#include "blade/runner.hh"

using namespace Blade;

int main() {
    Logger guard{};

    BL_INFO("Testing beamformer with the ATA kernel.");

    std::size_t numberOfWorkers = 2;
    std::vector<Vector<Device::CPU, CF32>> inputs(numberOfWorkers);
    std::vector<Vector<Device::CPU, CF32>> phasors(numberOfWorkers);
    std::vector<Vector<Device::CPU, CF32>> outputs(numberOfWorkers);

    auto runner = Blade::Runner<Test<CF32, CF32>>(numberOfWorkers, {
        .dims = {
            .NBEAMS = 16,
            .NANTS  = 20,
            .NCHANS = 192,
            .NTIME  = 8192,
            .NPOLS  = 2,
        },
        .blockSize = 512,
    });

    const auto& worker = runner.getWorker();
    for (std::size_t i = 0; i < numberOfWorkers; i++) {
        inputs[i].resize(worker.getInputSize());
        phasors[i].resize(worker.getPhasorsSize());
        outputs[i].resize(worker.getOutputSize());
    }

    for (int i = 0; i < 50; i++) {
        runner.enqueue([&](auto& worker){
            const auto& head = runner.getHead();
            worker.run(inputs[head], phasors[head], outputs[head]);
            return i;
        });

        std::size_t id;
        if (runner.dequeue(id)) {
            BL_INFO("Task {} finished.", id);
        }
    }

    BL_INFO("Test succeeded.");

    return 0;
}
