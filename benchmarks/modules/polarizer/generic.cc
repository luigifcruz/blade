#include "./generic.hh"

using namespace Blade;
namespace bm = benchmark;

// CF32 -> CF32

static void BM_Polarizer_Compute_CF32_CF32(bm::State& state) {
    ModuleUnderTest<Modules::Polarizer, CF32, CF32> mud;
    BL_CHECK_THROW(mud.runComputeBenchmark(state));
}

BENCHMARK(BM_Polarizer_Compute_CF32_CF32)
    ->Iterations(2<<13)
    ->Args({ 2, 1})
    ->Args({16, 1})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

static void BM_Polarizer_Transfer_CF32_CF32(bm::State& state) {
    ModuleUnderTest<Modules::Polarizer, CF32, CF32> mud;
    BL_CHECK_THROW(mud.runTransferBenchmark(state));
}

BENCHMARK(BM_Polarizer_Transfer_CF32_CF32)
    ->Iterations(64)
    ->Args({ 2, 1})
    ->Args({16, 1})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

static void BM_Polarizer_Converged_CF32_CF32(bm::State& state) {
    ModuleUnderTest<Modules::Polarizer, CF32, CF32> mud;
    BL_CHECK_THROW(mud.runConvergedBenchmark(state));
}

BENCHMARK(BM_Polarizer_Converged_CF32_CF32)
    ->Iterations(64)
    ->Args({ 2, 1})
    ->Args({16, 1})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

// CF16 -> CF16

static void BM_Polarizer_Compute_CF16_CF16(bm::State& state) {
    ModuleUnderTest<Modules::Polarizer, CF16, CF16> mud;
    BL_CHECK_THROW(mud.runComputeBenchmark(state));
}

BENCHMARK(BM_Polarizer_Compute_CF16_CF16)
    ->Iterations(2<<13)
    ->Args({ 2, 1})
    ->Args({16, 1})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

static void BM_Polarizer_Transfer_CF16_CF16(bm::State& state) {
    ModuleUnderTest<Modules::Polarizer, CF16, CF16> mud;
    BL_CHECK_THROW(mud.runTransferBenchmark(state));
}

BENCHMARK(BM_Polarizer_Transfer_CF16_CF16)
    ->Iterations(64)
    ->Args({ 2, 1})
    ->Args({16, 1})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

static void BM_Polarizer_Converged_CF16_CF16(bm::State& state) {
    ModuleUnderTest<Modules::Polarizer, CF16, CF16> mud;
    BL_CHECK_THROW(mud.runConvergedBenchmark(state));
}

BENCHMARK(BM_Polarizer_Converged_CF16_CF16)
    ->Iterations(64)
    ->Args({ 2, 1})
    ->Args({16, 1})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK_MAIN();
