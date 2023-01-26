#include "blade/modules/cast.hh"

#include "./generic.hh"

using namespace Blade;
namespace bm = benchmark;

// CF32 to CF32

static void BM_Cast_Compute_CF32_CF32(bm::State& state) {
    ModuleUnderTest<Modules::Cast, CF32, CF32> mud;
    BL_CHECK_THROW(mud.runComputeBenchmark(state));
}

BENCHMARK(BM_Cast_Compute_CF32_CF32)
    ->Iterations(2<<13)
    ->Args({2})
    ->Args({16})
    ->Args({64})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

static void BM_Cast_Transfer_CF32_CF32(bm::State& state) {
    ModuleUnderTest<Modules::Cast, CF32, CF32> mud;
    BL_CHECK_THROW(mud.runTransferBenchmark(state));
}

BENCHMARK(BM_Cast_Transfer_CF32_CF32)
    ->Iterations(64)
    ->Args({2})
    ->Args({16})
    ->Args({64})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

static void BM_Cast_Converged_CF32_CF32(bm::State& state) {
    ModuleUnderTest<Modules::Cast, CF32, CF32> mud;
    BL_CHECK_THROW(mud.runConvergedBenchmark(state));
}

BENCHMARK(BM_Cast_Converged_CF32_CF32)
    ->Iterations(64)
    ->Args({2})
    ->Args({16})
    ->Args({64})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

// CI8 to CF32

static void BM_Cast_Compute_CI8_CF32(bm::State& state) {
    ModuleUnderTest<Modules::Cast, CI8, CF32> mud;
    BL_CHECK_THROW(mud.runComputeBenchmark(state));
}

BENCHMARK(BM_Cast_Compute_CI8_CF32)
    ->Iterations(2<<13)
    ->Args({2})
    ->Args({16})
    ->Args({64})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

static void BM_Cast_Transfer_CI8_CF32(bm::State& state) {
    ModuleUnderTest<Modules::Cast, CI8, CF32> mud;
    BL_CHECK_THROW(mud.runTransferBenchmark(state));
}

BENCHMARK(BM_Cast_Transfer_CI8_CF32)
    ->Iterations(64)
    ->Args({2})
    ->Args({16})
    ->Args({64})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

static void BM_Cast_Converged_CI8_CF32(bm::State& state) {
    ModuleUnderTest<Modules::Cast, CI8, CF32> mud;
    BL_CHECK_THROW(mud.runConvergedBenchmark(state));
}

BENCHMARK(BM_Cast_Converged_CI8_CF32)
    ->Iterations(64)
    ->Args({2})
    ->Args({16})
    ->Args({64})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK_MAIN();
