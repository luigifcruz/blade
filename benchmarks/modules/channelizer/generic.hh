#include "blade/modules/channelizer/base.hh"

#include "./base.hh"

using namespace Blade;
namespace bm = benchmark;

static void BM_Channelizer_Compute(bm::State& state) {
    ChannelizerTest<Modules::Channelizer, CF32, CF32> mud;
    BL_CHECK_THROW(mud.run(state));
}

BENCHMARK(BM_Channelizer_Compute)
    ->Iterations(2<<13)
    ->Args({16, 8192})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);