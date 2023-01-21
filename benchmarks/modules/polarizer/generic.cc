#include "./generic.hh"

using namespace Blade;
namespace bm = benchmark;

static void BM_Polarizer_Compute(bm::State& state) {
    ModuleUnderTest<Modules::Polarizer, CF32, CF32> mud;
    BL_CHECK_THROW(mud.runComputeBenchmark(state));
}

BENCHMARK(BM_Polarizer_Compute)
    ->Iterations(2<<13)
    ->Args({ 2, 1})
    ->Args({16, 1})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

static void BM_Polarizer_Transfer(bm::State& state) {
    ModuleUnderTest<Modules::Polarizer, CF32, CF32> mud;
    BL_CHECK_THROW(mud.runTransferBenchmark(state));
}

BENCHMARK(BM_Polarizer_Transfer)
    ->Iterations(64)
    ->Args({ 2, 1})
    ->Args({16, 1})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

static void BM_Polarizer_Converged(bm::State& state) {
    ModuleUnderTest<Modules::Polarizer, CF32, CF32> mud;
    BL_CHECK_THROW(mud.runConvergedBenchmark(state));
}

BENCHMARK(BM_Polarizer_Converged)
    ->Iterations(64)
    ->Args({ 2, 1})
    ->Args({16, 1})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK_MAIN();
