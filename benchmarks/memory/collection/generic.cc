#include "./generic.hh"

using namespace Blade;
namespace bm = benchmark;

static void BM_Compute_AccumulationSOL_CF32(bm::State& state) {
    CollectionTest<CF32> mud;
    BL_CHECK_THROW(mud.runAccumulateSOLBenchmark(state));
}

BENCHMARK(BM_Compute_AccumulationSOL_CF32)
    ->Iterations(2<<13)
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

static void BM_Compute_Accumulation_CF32(bm::State& state) {
    CollectionTest<CF32> mud;
    BL_CHECK_THROW(mud.runAccumulateBenchmark(state));
}

BENCHMARK(BM_Compute_Accumulation_CF32)
    ->Iterations(2<<13)
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

static void BM_Compute_AccumulationMemcpy_CF32(bm::State& state) {
    CollectionTest<CF32> mud;
    BL_CHECK_THROW(mud.runAccumulateMemcpyBenchmark(state));
}

BENCHMARK(BM_Compute_AccumulationMemcpy_CF32)
    ->Iterations(2<<13)
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK_MAIN();
