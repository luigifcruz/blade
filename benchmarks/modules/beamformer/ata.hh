#include "blade/modules/beamformer/ata.hh"

#include "./base.hh"

using namespace Blade;
namespace bm = benchmark;

static void BM_BeamformerATA_Compute(bm::State& state) {
    BeamformerTest<Modules::Beamformer::ATA, CF32, CF32> mud;
    BL_CHECK_THROW(mud.run(state));
}

BENCHMARK(BM_BeamformerATA_Compute)
    ->Iterations(2<<13)
    ->Args({1, 20})
    ->Args({2, 20})
    ->Args({8, 20})
    ->Args({1, 42})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);