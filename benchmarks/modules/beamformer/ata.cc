#include "blade/modules/beamformer/ata.hh"

#include "./generic.hh"

using namespace Blade;
namespace bm = benchmark;

static void BM_BeamformerATA_Compute(bm::State& state) {
    ModuleUnderTest<Modules::Beamformer::ATA, CF32, CF32> mud;
    BL_CHECK_THROW(mud.runComputeBenchmark(state));
}

BENCHMARK(BM_BeamformerATA_Compute)
    ->Iterations(2<<13)
    ->Args({1, 20})
    ->Args({2, 20})
    ->Args({8, 20})
    ->Args({1, 42})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

static void BM_BeamformerATA_Transfer(bm::State& state) {
    ModuleUnderTest<Modules::Beamformer::ATA, CF32, CF32> mud;
    BL_CHECK_THROW(mud.runTransferBenchmark(state));
}

// TODO: Remove Transfer and Converged Benchmarks of Modules.
BENCHMARK(BM_BeamformerATA_Transfer)
    ->Iterations(64)
    ->Args({1, 20})
    ->Args({1, 42})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

static void BM_BeamformerATA_Converged(bm::State& state) {
    ModuleUnderTest<Modules::Beamformer::ATA, CF32, CF32> mud;
    BL_CHECK_THROW(mud.runConvergedBenchmark(state));
}

BENCHMARK(BM_BeamformerATA_Converged)
    ->Iterations(64)
    ->Args({1, 20})
    ->Args({2, 20})
    ->Args({8, 20})
    ->Args({1, 42})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK_MAIN();
