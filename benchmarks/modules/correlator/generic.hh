#include "./base.hh"

#include "blade/modules/correlator.hh"

using namespace Blade;
namespace bm = benchmark;

static void BM_Correlator_Compute(bm::State& state) {
    CorrelatorTest<Modules::Correlator, CI8, CF32> mud;
    BL_CHECK_THROW(mud.run(state));
}

// ATA Standard Mode
// TODO: [GLOB, SHAMEM], IT[CI8, CF32], XT[CI8, CF32, CF64]

BENCHMARK(BM_Correlator_Compute)
    ->Iterations(2<<8)
    ->Args({28, 192, 8192, 2, 1, 64})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

// ATA High-Resolution Mode

BENCHMARK(BM_Correlator_Compute)
    ->Iterations(2<<8)
    ->Args({28, 65536, 1, 2, 1, 32})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);
