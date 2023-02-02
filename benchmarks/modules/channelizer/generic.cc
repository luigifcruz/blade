#include "blade/modules/channelizer/base.hh"

#include "./generic.hh"

using namespace Blade;
namespace bm = benchmark;

static void BM_Channelizer_Compute(bm::State& state) {
    ModuleUnderTest<Modules::Channelizer, CF32, CF32> mud;
    BL_CHECK_THROW(mud.runComputeBenchmark(state));
}

BENCHMARK(BM_Channelizer_Compute)
    ->Iterations(2<<13)
    ->Args({16, 8192})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

static void BM_Channelizer_Transfer(bm::State& state) {
    ModuleUnderTest<Modules::Channelizer, CF32, CF32> mud;
    BL_CHECK_THROW(mud.runTransferBenchmark(state));
}

BENCHMARK(BM_Channelizer_Transfer)
    ->Iterations(64)
    ->Args({16, 8192})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

static void BM_Channelizer_Converged(bm::State& state) {
    ModuleUnderTest<Modules::Channelizer, CF32, CF32> mud;
    BL_CHECK_THROW(mud.runConvergedBenchmark(state));
}

BENCHMARK(BM_Channelizer_Converged)
    ->Iterations(64)
    ->Args({16, 8192})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK_MAIN();
