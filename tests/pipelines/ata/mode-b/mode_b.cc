#include <memory>
#include <cassert>

#include "blade/base.hh"
#include "blade/logger.hh"
#include "blade/runner.hh"
#include "blade/plan.hh"
#include "blade/pipelines/ata/mode_b.hh"

extern "C" {
#include "mode_b.h"
}

using namespace Blade;
using namespace Blade::Pipelines::ATA;

using TestPipeline = ModeB<BLADE_ATA_MODE_B_OUTPUT_ELEMENT_T>;

static std::unique_ptr<Runner<TestPipeline>> runner;
static Vector<Device::CPU, F64> dummyJulianDate({1});
static Vector<Device::CPU, F64> dummyDut1({1});
static Vector<Device::CPU, U64> dummyFrequencyChannelOffset({1});

bool blade_use_device(int device_id) {
    return SetCudaDevice(device_id) == Result::SUCCESS;
}

bool blade_ata_b_initialize(U64 numberOfWorkers) {
    if (runner) {
        BL_FATAL("Can't initialize because Blade Runner is already initialized.");
        BL_CHECK_THROW(Result::ASSERTION_ERROR);
    }

    dummyJulianDate[0] = (1649366473.0 / 86400) + 2440587.5;
    dummyDut1[0] = 0.0;
    dummyFrequencyChannelOffset[0] = 0;

    runner = Runner<TestPipeline>::New(numberOfWorkers, {
        .inputDimensions = {
            .A = BLADE_ATA_MODE_B_NANT,
            .F = BLADE_ATA_MODE_B_NCHAN,
            .T = BLADE_ATA_MODE_B_NTIME,
            .P = BLADE_ATA_MODE_B_NPOL,
        },

        .preBeamformerChannelizerRate = BLADE_ATA_MODE_B_CHANNELIZER_RATE,

        .phasorObservationFrequencyHz = 6500.125*1e6,
        .phasorChannelBandwidthHz = 0.5e6,
        .phasorTotalBandwidthHz = 1.024e9,
        .phasorFrequencyStartIndex = 352,
        .phasorReferenceAntennaIndex = 0,
        .phasorArrayReferencePosition = {
            .LON = BL_DEG_TO_RAD(-121.470733), 
            .LAT = BL_DEG_TO_RAD(40.815987),
            .ALT = 1020.86,
        },
        .phasorBoresightCoordinate = {
            .RA = 0.64169,
            .DEC = 1.079896295,
        },
        .phasorAntennaPositions = {
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
        .phasorAntennaCoefficients = std::vector<CF64>(ArrayDimensions({
            BLADE_ATA_MODE_B_NANT,
            BLADE_ATA_MODE_B_NCHAN * BLADE_ATA_MODE_B_CHANNELIZER_RATE,
            1,
            BLADE_ATA_MODE_B_NPOL,
        }).size()),
        .phasorBeamCoordinates = {
            {0.63722, 1.07552424},
            {0.64169, 1.079896295},
            {0.64169, 1.079896295},
            {0.64169, 1.079896295},
            {0.64169, 1.079896295},
            {0.64169, 1.079896295},
            {0.64169, 1.079896295},
            {0.64169, 1.079896295},
        },

        .beamformerIncoherentBeam = BLADE_ATA_MODE_B_ENABLE_INCOHERENT_BEAM,

        .detectorEnable = BLADE_ATA_MODE_B_DETECTOR_ENABLED,
        .detectorIntegrationSize = BLADE_ATA_MODE_B_DETECTOR_INTEGRATION,
        .detectorNumberOfOutputPolarizations = BLADE_ATA_MODE_B_DETECTOR_POLS,
    });

    return true;
}

void blade_ata_b_terminate() {
    if (!runner) {
        BL_FATAL("Can't terminate because Blade Runner isn't initialized.");
        BL_CHECK_THROW(Result::ASSERTION_ERROR);
    }
    runner.reset();
}

U64 blade_ata_b_get_input_size() {
    assert(runner);
    return runner->getWorker().getInputBuffer().size();
}

U64 blade_ata_b_get_output_size() {
    assert(runner);
    return runner->getWorker().getOutputBuffer().size();
}

bool blade_pin_memory(void* buffer, U64 size) {
    return Memory::PageLock(Vector<Device::CPU, U8>(buffer, {size})) == Result::SUCCESS;
}

bool blade_ata_b_enqueue(void* input_ptr, void* output_ptr, U64 id) {
    assert(runner);

    return runner->enqueue([&](auto& worker) {
        // Convert C pointers to Blade::Vector.
        auto input = ArrayTensor<Device::CPU, CI8>(input_ptr, worker.getInputBuffer().dims());
        auto output = ArrayTensor<Device::CPU, BLADE_ATA_MODE_B_OUTPUT_ELEMENT_T>(output_ptr, 
                worker.getOutputBuffer().dims());

        // Transfer input data from CPU memory to the worker.
        Plan::TransferIn(worker, dummyJulianDate, dummyDut1, dummyFrequencyChannelOffset, input);

        // Compute block.
        Plan::Compute(worker);

        // Transfer output data from the worker to the CPU memory.
        Plan::TransferOut(output, worker.getOutputBuffer(), worker);

        return id;
    });
}

bool blade_ata_b_dequeue(U64* id) {
    assert(runner);
    return runner->dequeue(id);
}
