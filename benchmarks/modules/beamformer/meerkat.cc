#include "blade/modules/beamformer/meerkat.hh"

// TODO: Add more realistic MeerKAT/VLA input dimensions.

#include "./generic.hh"

using namespace Blade;
namespace bm = benchmark;

static void BM_BeamformerMEERKAT_Compute(bm::State& state) {
    ModuleUnderTest<Modules::Beamformer::MeerKAT, CF32, CF32> mud;
    BL_CHECK_THROW(mud.runComputeBenchmark(state));
}

BENCHMARK(BM_BeamformerMEERKAT_Compute)
    ->Iterations(2<<13)
    ->Args({1, 20})
    ->Args({2, 20})
    ->Args({8, 20})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

static void BM_BeamformerMEERKAT_Transfer(bm::State& state) {
    ModuleUnderTest<Modules::Beamformer::MeerKAT, CF32, CF32> mud;
    BL_CHECK_THROW(mud.runTransferBenchmark(state));
}

BENCHMARK(BM_BeamformerMEERKAT_Transfer)
    ->Iterations(64)
    ->Args({1, 20})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

static void BM_BeamformerMEERKAT_Converged(bm::State& state) {
    ModuleUnderTest<Modules::Beamformer::MeerKAT, CF32, CF32> mud;
    BL_CHECK_THROW(mud.runConvergedBenchmark(state));
}

BENCHMARK(BM_BeamformerMEERKAT_Converged)
    ->Iterations(64)
    ->Args({1, 20})
    ->Args({2, 20})
    ->Args({8, 20})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK_MAIN();
