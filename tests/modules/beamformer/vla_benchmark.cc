#include "pipeline_vla.hh"
#include "blade/modules/beamformer/vla.hh"

#include <benchmark/benchmark.h>

using namespace Blade;

static void BEAMFORM(benchmark::State& state) {
    BL_INFO("Benchmarking the beamformer with the VLA kernel.");

    Test<CF32, CF32> mod({
        .numberOfBeams = U64(state.range(0)),
        .numberOfAntennas = 27,
        .numberOfFrequencyChannels = 32,
        .numberOfTimeSamples = 8192,
        .numberOfPolarizations = 2,
        .blockSize = 512,
    });

    Vector<Device::CPU, CF32> input(mod.getInputSize());
    Vector<Device::CPU, CF32> phasors(mod.getPhasorsSize());
    Vector<Device::CPU, CF32> output(mod.getOutputSize());

    if (mod.run(input, phasors, output, true) != Result::SUCCESS) {
        BL_WARN("Fault was encountered...");
    }
    
    for (auto _ : state) {
        mod.run(input, phasors, output, true);
    }
    state.SetBytesProcessed(
      int64_t(state.iterations())*mod.getInputSize()*sizeof(CF32)
    );
}

BENCHMARK(BEAMFORM)->Iterations(100)->DenseRange(1, 33, 2);

BENCHMARK_MAIN();
