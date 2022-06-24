#include "blade/base.hh"
#include "blade/logger.hh"
#include "blade/runner.hh"
#include "blade/pipelines/vla/mode_b.hh"

#include <benchmark/benchmark.h>

using namespace Blade;
using namespace Blade::Pipelines::VLA;

static void PIPELINE(benchmark::State& state) {
    BL_INFO("Benchmarking the VLA Mode B Pipeline.");
    
    using BenchPipeline = ModeB<CF32>;
    BenchPipeline::Config config = {
        .numberOfAntennas = 27,
        .numberOfFrequencyChannels = 32,
        .numberOfTimeSamples = 8192,
        .numberOfPolarizations = 2,

        .channelizerRate = 8192/512,

        .beamformerBeams = U64(state.range(0)),

        .outputMemWidth = 8192,
        .outputMemPad = 0,

        .castBlockSize = 512,
        .channelizerBlockSize = 512,
        .phasorsBlockSize = 512,
        .beamformerBlockSize = 512,
    };
    const int numberOfWorkers = 2;
    Runner<BenchPipeline> runner = Runner<BenchPipeline>(numberOfWorkers, config);

    Vector<Device::CPU, CI8> *input_buffers[numberOfWorkers];
    Vector<Device::CPU, CF32> *phasors_buffers[numberOfWorkers];
    Vector<Device::CPU, CF32> *output_buffers[numberOfWorkers];

    for (int i = 0; i < numberOfWorkers; i++) {
        input_buffers[i] = new Vector<Device::CPU, CI8>(runner.getWorker().getInputSize());
        phasors_buffers[i] = new Vector<Device::CPU, CF32>(runner.getWorker().getPhasorsSize());
        output_buffers[i] = new Vector<Device::CPU, CF32>(runner.getWorker().getOutputSize());
    }

    int buffer_idx = 0, job_idx = 0;
    U64 dequeue_id;
    for (auto _ : state) {
      if (runner.enqueue(
        [&](auto& worker){
          worker.run(*input_buffers[buffer_idx], *phasors_buffers[buffer_idx], *output_buffers[buffer_idx]);
          return job_idx;
        }
      )) {
        buffer_idx = (buffer_idx + 1) % numberOfWorkers;
      }

      if (runner.dequeue(&dequeue_id)) {
        job_idx++;
      }
    }
    state.SetBytesProcessed(
      int64_t(state.iterations())*runner.getWorker().getInputSize()*sizeof(CI8)
    );
}

BENCHMARK(PIPELINE)->Iterations(100)->DenseRange(1, 16, 4);

BENCHMARK_MAIN();
