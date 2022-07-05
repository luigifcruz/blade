#include "blade/base.hh"
#include "blade/logger.hh"
#include "blade/runner.hh"
#include "blade/pipelines/ata/mode_b.hh"

#include <benchmark/benchmark.h>

using namespace Blade;
using namespace Blade::Pipelines::ATA;

static void PIPELINE(benchmark::State& state) {
    BL_INFO("Benchmarking the ATA Mode B Pipeline.");
    
    U64 numberOfBeams = U64(state.range(0));
    std::vector<RA_DEC> beamCoordinates = std::vector<RA_DEC>(numberOfBeams);

    using BenchPipeline = ModeB<CF32>;
    BenchPipeline::Config config = {
        .numberOfAntennas = 20,
        .numberOfFrequencyChannels = 192,
        .numberOfTimeSamples = 16384,
        .numberOfPolarizations = 2,

        .channelizerRate = 1,

        .beamformerBeams = numberOfBeams,

        .rfFrequencyHz = 6500.125*1e6,
        .channelBandwidthHz = 0.5e6,
        .totalBandwidthHz = 1.024e9,
        .frequencyStartIndex = 352,
        .referenceAntennaIndex = 0,
        .arrayReferencePosition = {
            .LON = BL_DEG_TO_RAD(-121.470733), 
            .LAT = BL_DEG_TO_RAD(40.815987),
            .ALT = 1020.86,
        },
        .boresightCoordinate = {
            .RA = 0.64169,
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
        .beamCoordinates = beamCoordinates,

        .outputMemWidth = 8192,
        .outputMemPad = 0,

        .castBlockSize = 512,
        .channelizerBlockSize = 512,
        .phasorsBlockSize = 512,
        .beamformerBlockSize = 512,
    };
    config.antennaCalibrations.resize(
        config.numberOfAntennas *
        config.numberOfFrequencyChannels *
        config.channelizerRate *
        config.numberOfPolarizations);

    const int numberOfWorkers = 2;
    Runner<BenchPipeline> runner = Runner<BenchPipeline>(numberOfWorkers, config);

    Vector<Device::CPU, CI8> *input_buffers[numberOfWorkers];
    Vector<Device::CPU, CF32> *output_buffers[numberOfWorkers];

    for (int i = 0; i < numberOfWorkers; i++) {
        input_buffers[i] = new Vector<Device::CPU, CI8>(runner.getWorker().getInputSize());
        output_buffers[i] = new Vector<Device::CPU, CF32>(runner.getWorker().getOutputSize());
    }

    F64 time_jd, dut1;

    int buffer_idx = 0, job_idx = 0;
    U64 dequeue_id;
    for (auto _ : state) {
      if (runner.enqueue(
        [&](auto& worker){
          worker.run(
            time_jd, dut1,
            *input_buffers[buffer_idx],
            *output_buffers[buffer_idx]);
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

BENCHMARK(PIPELINE)->Iterations(500)->DenseRange(1, 33, 4);

BENCHMARK_MAIN();
