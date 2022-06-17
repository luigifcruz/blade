#include <memory>
#include <cassert>

#include "blade/base.hh"
#include "blade/logger.hh"
#include "blade/runner.hh"
#include "blade/pipelines/vla/mode_b.hh"

extern "C" {
#include "mode_b.h"
}

using namespace Blade;
using namespace Blade::Pipelines::VLA;

using TestPipeline = ModeB<BLADE_VLA_MODE_B_OUTPUT_ELEMENT_T>;

static std::unique_ptr<Runner<TestPipeline>> runner;

bool blade_use_device(int device_id) {
    return SetCudaDevice(device_id) == Result::SUCCESS;
}

bool blade_vla_b_initialize(U64 numberOfWorkers) {
    if (runner) {
        BL_FATAL("Can't initialize because Blade Runner is already initialized.");
        throw Result::ASSERTION_ERROR;
    }

    TestPipeline::Config config = {
        .numberOfAntennas = BLADE_VLA_MODE_B_INPUT_NANT,
        .numberOfFrequencyChannels = BLADE_VLA_MODE_B_ANT_NCHAN,
        .numberOfTimeSamples = BLADE_VLA_MODE_B_NTIME,
        .numberOfPolarizations = BLADE_VLA_MODE_B_NPOL,

        .channelizerRate = BLADE_VLA_MODE_B_CHANNELIZER_RATE,

        .beamformerBeams = BLADE_VLA_MODE_B_OUTPUT_NBEAM,

        .outputMemWidth = BLADE_VLA_MODE_B_OUTPUT_MEMCPY2D_WIDTH,
        .outputMemPad = BLADE_VLA_MODE_B_OUTPUT_MEMCPY2D_PAD,

        .castBlockSize = 512,
        .channelizerBlockSize = 512,
        .phasorsBlockSize = 512,
        .beamformerBlockSize = 512,
    };

    runner = Runner<TestPipeline>::New(numberOfWorkers, config);

    return true;
}

void blade_vla_b_terminate() {
    if (!runner) {
        BL_FATAL("Can't terminate because Blade Runner isn't initialized.");
        throw Result::ASSERTION_ERROR;
    }
    runner.reset();
}

U64 blade_vla_b_get_input_size() {
    assert(runner);
    return runner->getWorker().getInputSize();
}

U64 blade_vla_b_get_phasors_size() {
    assert(runner);
    return runner->getWorker().getPhasorsSize();
}

U64 blade_vla_b_get_output_size() {
    assert(runner);
    return runner->getWorker().getOutputSize();
}

bool blade_pin_memory(void* buffer, U64 size) {
    return Memory::PageLock(Vector<Device::CPU, I8>(buffer, size)) == Result::SUCCESS;
}

bool blade_vla_b_enqueue(void* input_ptr, void* phasor_ptr, void* output_ptr, U64 id) {
    assert(runner);
    return runner->enqueue([&](auto& worker){
        auto input = Vector<Device::CPU, CI8>(input_ptr, worker.getInputSize());
        auto phasor = Vector<Device::CPU, CF32>(phasor_ptr, worker.getPhasorsSize());
        auto output = Vector<Device::CPU, BLADE_VLA_MODE_B_OUTPUT_ELEMENT_T>
            (output_ptr, worker.getOutputSize());

        worker.run(input, phasor, output);

        return id;
    });
}

bool blade_vla_b_dequeue(U64* id) {
    assert(runner);
    return runner->dequeue(id);
}
