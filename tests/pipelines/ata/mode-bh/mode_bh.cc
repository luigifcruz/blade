#include <memory>
#include <cassert>

#include "blade/plan.hh"
#include "blade/base.hh"
#include "blade/logger.hh"
#include "blade/runner.hh"
#include "blade/pipelines/ata/mode_b.hh"
#include "blade/pipelines/generic/mode_h.hh"

extern "C" {
#include "mode_bh.h"
}

using namespace Blade;
using namespace Blade::Pipelines::ATA;
using namespace Blade::Pipelines::Generic;

using TestPipelineB = ModeB<CI8, CF32>;
using TestPipelineH = ModeH<CF32, F32>;

static struct {
    U64 StepCount = 0;
    void* UserData = nullptr;
    std::unordered_map<U64, void*> InputPointerMap;
    std::unordered_map<U64, void*> OutputPointerMap;

    struct {
        std::unique_ptr<Runner<TestPipelineB>> B; 
        std::unique_ptr<Runner<TestPipelineH>> H; 
    } RunnersInstances;

    struct {
        blade_input_buffer_fetch_cb* InputBufferFetch;
        blade_input_buffer_ready_cb* InputBufferReady;
        blade_output_buffer_fetch_cb* OutputBufferFetch;
        blade_output_buffer_ready_cb* OutputBufferReady;
    } Callbacks;
} State;

static Vector<Device::CPU, F64> dummyJulianDate({1});
static Vector<Device::CPU, F64> dummyDut1({1});
static Vector<Device::CPU, U64> dummyFrequencyChannelOffset({1});

bool blade_pin_memory(void* buffer, U64 size) {
    return Memory::PageLock(Vector<Device::CPU, U8>(buffer, {size})) == Result::SUCCESS;
}

bool blade_use_device(int device_id) {
    return SetCudaDevice(device_id) == Result::SUCCESS;
}

bool blade_ata_bh_initialize(U64 numberOfWorkers) {
    if (State.RunnersInstances.B || State.RunnersInstances.H) {
        BL_FATAL("Can't initialize because Blade Runner is already initialized.");
        BL_CHECK_THROW(Result::ASSERTION_ERROR);
    }

    dummyJulianDate[0] = (1649366473.0 / 86400) + 2440587.5;
    dummyDut1[0] = 0.0;
    dummyFrequencyChannelOffset[0] = 0;

    State.RunnersInstances.B = Runner<TestPipelineB>::New(numberOfWorkers, {
        .inputDimensions = {
            .A = BLADE_ATA_MODE_BH_NANT,
            .F = BLADE_ATA_MODE_BH_NCHAN,
            .T = BLADE_ATA_MODE_BH_NTIME,
            .P = BLADE_ATA_MODE_BH_NPOL,
        },

        .preBeamformerChannelizerRate = BLADE_ATA_MODE_BH_CHANNELIZER_RATE,

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
            {-2523824.598229116, -4123527.93080514, 4147833.98936114},        // 5b
        },
        .phasorAntennaCoefficients = std::vector<CF64>(ArrayDimensions({
            BLADE_ATA_MODE_BH_NANT,
            BLADE_ATA_MODE_BH_NCHAN * BLADE_ATA_MODE_BH_CHANNELIZER_RATE,
            1,
            BLADE_ATA_MODE_BH_NPOL,
        }).size()),
        .phasorBeamCoordinates = {
            {0.64169, 1.079896295},
            {0.64169, 1.079896295},
        },
        .phasorAntennaCoefficientChannelRate = BLADE_ATA_MODE_BH_CHANNELIZER_RATE,
    });

    State.RunnersInstances.H = Runner<TestPipelineH>::New(numberOfWorkers, {
        .inputDimensions = State.RunnersInstances.B->getWorker().getOutputBuffer().dims(),

        .accumulateRate = BLADE_ATA_MODE_BH_ACCUMULATE_RATE, 

        .detectorIntegrationSize = BLADE_ATA_MODE_BH_INTEGRATION_SIZE,
        .detectorKernel = DetectorKernel::AFTP_1pol,
    });

    State.InputPointerMap.reserve(numberOfWorkers);
    State.OutputPointerMap.reserve(numberOfWorkers);

    return true;
}

void blade_ata_bh_terminate() {
    if (!State.RunnersInstances.B || !State.RunnersInstances.H) {
        BL_FATAL("Can't terminate because Blade Runner isn't initialized.");
        BL_CHECK_THROW(Result::ASSERTION_ERROR);
    }

    State.RunnersInstances.B.reset();
    State.RunnersInstances.H.reset();
}

U64 blade_ata_bh_get_input_size() {
    assert(State.RunnersInstances.B);
    return State.RunnersInstances.B->getWorker().getInputBuffer().size();
}

U64 blade_ata_bh_get_output_size() {
    assert(State.RunnersInstances.H);
    return State.RunnersInstances.H->getWorker().getOutputBuffer().size();
}

void blade_ata_bh_register_user_data(void* user_data) {
    State.UserData = user_data;
}

void blade_ata_bh_register_input_buffer_fetch_cb(blade_input_buffer_fetch_cb* f) {
    State.Callbacks.InputBufferFetch = f;
}

void blade_ata_bh_register_input_buffer_ready_cb(blade_input_buffer_ready_cb* f) {
    State.Callbacks.InputBufferReady = f;
}

void blade_ata_bh_register_output_buffer_fetch_cb(blade_output_buffer_fetch_cb* f) {
    State.Callbacks.OutputBufferFetch = f;
}

void blade_ata_bh_register_output_buffer_ready_cb(blade_output_buffer_ready_cb* f) {
    State.Callbacks.OutputBufferReady = f;
}

void blade_ata_bh_compute_step() {
    assert(State.RunnersInstances.B);
    assert(State.RunnersInstances.H);

    U64 callbackStep = 0;
    void* externalBuffer = nullptr;

    auto& ModeB = State.RunnersInstances.B; 
    auto& ModeH = State.RunnersInstances.H; 

    ModeB->enqueue([&](auto& worker) {
        // Check if next runner has free slot.
        Plan::Available(ModeH);

        // Calls client callback to request empty input buffer.
        if (!State.Callbacks.InputBufferFetch(State.UserData, &externalBuffer)) {
            Plan::Skip();
        }

        // Keeps track of pointer for "ready" callback.
        State.InputPointerMap.insert({State.StepCount, externalBuffer});

        // Create Memory::Vector from RAW pointer.
        auto input = ArrayTensor<Device::CPU, CI8>(externalBuffer, worker.getInputBuffer().dims());

        // Transfer input memory to the pipeline.
        Plan::TransferIn(worker, 
                         dummyJulianDate,
                         dummyDut1,
                         dummyFrequencyChannelOffset,
                         input);

        // Compute input data.
        Plan::Compute(worker);

        // Concatenate output data inside next pipeline input buffer.
        Plan::Accumulate(ModeH, ModeB, worker.getOutputBuffer());

        // Return job identity and increment counter.
        return State.StepCount++; 
    });

    ModeH->enqueue([&](auto& worker) {
        // Try dequeue job from last runner. If unlucky, return.
        Plan::Dequeue(ModeB, &callbackStep);

        // If dequeue successfull, recycle input buffer.
        const auto& recycleBuffer = State.InputPointerMap[callbackStep];
        State.Callbacks.InputBufferReady(State.UserData, recycleBuffer);
        State.InputPointerMap.erase(callbackStep);

        // Compute input data.
        Plan::Compute(worker);

        // Calls client callback to request empty output buffer.
        if (!State.Callbacks.OutputBufferFetch(State.UserData, &externalBuffer)) {
            Plan::Skip();
        }

        // Keeps track of pointer for "ready" callback.
        State.OutputPointerMap.insert({callbackStep, externalBuffer});

        // Create Memory::Vector from RAW pointer.
        auto output = ArrayTensor<Device::CPU, F32>(externalBuffer, worker.getOutputBuffer().dims());

        // Copy worker output to external output buffer.
        Plan::TransferOut(output, worker.getOutputBuffer(), worker);

        // Return job identity.
        return callbackStep;
    });

    // Dequeue last runner job and recycle output buffer.
    if (ModeH->dequeue(&callbackStep)) {
        const auto& recycleBuffer = State.OutputPointerMap[callbackStep];
        State.Callbacks.OutputBufferReady(State.UserData, recycleBuffer);
        State.OutputPointerMap.erase(callbackStep);
    }

    // Prevent memory clobber inside spin-loop.
    Plan::Loop();
}
