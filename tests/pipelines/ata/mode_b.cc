#include <memory>

#include "blade/base.hh"
#include "blade/logger.hh"
#include "blade/runner.hh"
#include "blade/pipelines/ata/mode_b.hh"

extern "C" {
#include "mode_b.h"
}

using namespace Blade;
using namespace Blade::Pipelines::ATA;

using TestPipeline = ModeB<BLADE_ATA_MODE_B_OUTPUT_ELEMENT_T>;

static struct {
    std::unique_ptr<Logger> guard;
    std::unique_ptr<Runner<TestPipeline>> runner;
} instance;

bool blade_use_device(int device_id) {
    return SetCudaDevice(device_id) == Result::SUCCESS;
}

bool blade_ata_b_initialize(U64 numberOfWorkers) {
    TestPipeline::Config config = {
        .numberOfBeams = 1,
        .numberOfAntennas = BLADE_ATA_MODE_B_INPUT_NANT,
        .numberOfFrequencyChannels = BLADE_ATA_MODE_B_ANT_NCHAN,
        .numberOfTimeSamples = BLADE_ATA_MODE_B_NTIME,
        .numberOfPolarizations = BLADE_ATA_MODE_B_NPOL,

        .channelizerRate = BLADE_ATA_MODE_B_CHANNELIZER_RATE,

        .beamformerBeams = BLADE_ATA_MODE_B_OUTPUT_NBEAM,

        .rfFrequencyHz = 0,
        .channelBandwidthHz = 0,
        .totalBandwidthHz = 0,
        .frequencyStartIndex = 0,
        .referenceAntennaIndex = 0,
        .arrayReferencePosition = {
            .LON = BL_DEG_TO_RAD(-121.470733), 
            .LAT = BL_DEG_TO_RAD(40.815987),
            .ALT = 1020.86,
        },
        .boresightCoordinate = {
            .RA = 0.94169,
            .DEC = 1.079896295,
        },
        .antennaPositions = {
            {-2524041.5388905862, -4123587.965024342, 4147646.4222955606},    // 1c 
            {-2524068.187873109, -4123558.735413135, 4147656.21282186},       // 1e 
            {-2524087.2078100787, -4123532.397416349, 4147670.9866770394},    // 1g 
            {-2524103.384010733, -4123511.111598937, 4147682.4133068994},     // 1h 
            {-2524056.730228759, -4123515.287949227, 4147706.4850287656},     // 1k 
            {-2523986.279601761, -4123497.427940991, 4147766.732988923},      // 2a 
            {-2523970.301363642, -4123515.238502669, 4147758.790023165},      // 2b 
            {-2523983.5419911123, -4123528.1422073604, 4147737.872218138},    // 2c 
            {-2523941.5221860334, -4123568.125040547, 4147723.8292249846},    // 2e 
            {-2524074.096220788, -4123468.5182652213, 4147742.0422435375},    // 2h 
            {-2524058.6409591637, -4123466.5112451194, 4147753.4513993543},   // 2j 
            {-2524026.989692545, -4123480.9405167866, 4147758.2356800516},    // 2l 
            {-2524048.5254066754, -4123468.3463909747, 4147757.835369889},    // 2k 
            {-2524000.5641107005, -4123498.2984570004, 4147756.815976133},    // 2m 
            {-2523945.086670364, -4123480.3638816103, 4147808.127865142},     // 3d 
            {-2523950.6822576034, -4123444.7023326857, 4147839.7474427638},   // 3l 
            {-2523880.869769226, -4123514.3375464156, 4147813.413426994},     // 4e 
            {-2523930.3747946257, -4123454.3080821196, 4147842.6449955846},   // 4g 
            {-2523898.1150373477, -4123456.314794732, 4147860.3045849088},    // 4j 
            {-2523824.598229116, -4123527.93080514, 4147833.98936114}         // 5b
        },
        .antennaCalibrations = {},
        .beamCoordinates = {
            {0.64169, 1.079896295},
            {0.64169, 1.079896295},
            {0.64169, 1.079896295},
            {0.64169, 1.079896295},
            {0.64169, 1.079896295},
            {0.64169, 1.079896295},
            {0.64169, 1.079896295},
            {0.64169, 1.079896295},
        },

        .outputMemWidth = BLADE_ATA_MODE_B_OUTPUT_MEMCPY2D_WIDTH,
        .outputMemPad = BLADE_ATA_MODE_B_OUTPUT_MEMCPY2D_PAD,

        .castBlockSize = 512,
        .channelizerBlockSize = 512,
        .phasorsBlockSize = 512,
        .beamformerBlockSize = 512,
    };

    //  TODO: [RELEASE] REMOVE.
    config.antennaCalibrations.resize(20*768*2);

    instance.guard = std::make_unique<Logger>();
    instance.runner = Runner<TestPipeline>::New(numberOfWorkers, config);

    return true;
}

void blade_ata_b_terminate() {
    instance.runner.reset();
    instance.guard.reset();
}

U64 blade_ata_b_get_input_size() {
    assert(instance.runner);
    return instance.runner->getWorker().getInputSize();
}

U64 blade_ata_b_get_output_size() {
    assert(instance.runner);
    return instance.runner->getWorker().getOutputSize();
}

U64 blade_ata_b_get_phasor_size() {
    assert(instance.runner);
    return instance.runner->getWorker().getPhasorsSize();
}

bool blade_pin_memory(void* buffer, U64 size) {
    return Memory::PageLock(Vector<Device::CPU, I8>(buffer, size)) == Result::SUCCESS;
}

bool blade_ata_b_set_phasors(void* phasors, bool block) {
    assert(instance.runner);

    return instance.runner->applyToAllWorkers([&](auto& worker){
        const auto& size = worker.getPhasorsSize();
        return worker.setPhasors(Vector<Device::CPU, CF32>(phasors, size));
    }, block) == Result::SUCCESS;
}

bool blade_ata_b_enqueue(void* input_ptr, void* output_ptr, U64 id) {
    assert(instance.runner);
    return instance.runner->enqueue([&](auto& worker){
        auto input = Vector<Device::CPU, CI8>(input_ptr, worker.getInputSize());
        auto output = Vector<Device::CPU, BLADE_ATA_MODE_B_OUTPUT_ELEMENT_T>
            (output_ptr, worker.getOutputSize());

        worker.run(input, output);

        return id;
    });
}

bool blade_ata_b_dequeue(U64* id) {
    assert(instance.runner);
    return instance.runner->dequeue(id);
}
