#include "blade/modules/detector.hh"

#include "./generic.hh"

using namespace Blade;
namespace bm = benchmark;

static void BM_Detector_Compute(bm::State& state) {
    ModuleUnderTest<Modules::Detector, CF32, F32> mud;
    BL_CHECK_THROW(mud.runComputeBenchmark(state));
}

BENCHMARK(BM_Detector_Compute)
    ->Iterations(2<<13)
    ->Args({2,  8, 4})
    ->Args({16, 8, 4})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_Detector_Compute)
    ->Iterations(2<<13)
    ->Args({2,  32, 4})
    ->Args({16, 32, 4})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_Detector_Compute)
    ->Iterations(2<<13)
    ->Args({2,  64, 4})
    ->Args({16, 64, 4})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

static void BM_Detector_Transfer(bm::State& state) {
    ModuleUnderTest<Modules::Detector, CF32, F32> mud;
    BL_CHECK_THROW(mud.runTransferBenchmark(state));
}

BENCHMARK(BM_Detector_Transfer)
    ->Iterations(64)
    ->Args({2,  8, 4})
    ->Args({16, 8, 4})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

static void BM_Detector_Converged(bm::State& state) {
    ModuleUnderTest<Modules::Detector, CF32, F32> mud;
    BL_CHECK_THROW(mud.runConvergedBenchmark(state));
}

BENCHMARK(BM_Detector_Converged)
    ->Iterations(64)
    ->Args({2,  8, 4})
    ->Args({16, 8, 4})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK_MAIN();
