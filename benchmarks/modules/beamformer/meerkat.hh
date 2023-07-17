#include "blade/modules/beamformer/meerkat.hh"

// TODO: Add more realistic MeerKAT input dimensions.

#include "./base.hh"

using namespace Blade;
namespace bm = benchmark;

static void BM_BeamformerMEERKAT_Compute(bm::State& state) {
    BeamformerTest<Modules::Beamformer::MeerKAT, CF32, CF32> mud;
    BL_CHECK_THROW(mud.run(state));
}

BENCHMARK(BM_BeamformerMEERKAT_Compute)
    ->Iterations(2<<13)
    ->Args({1, 20})
    ->Args({2, 20})
    ->Args({8, 20})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);