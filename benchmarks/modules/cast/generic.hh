#include "blade/modules/cast.hh"

#include "./base.hh"

using namespace Blade;
namespace bm = benchmark;

// CF32 to CF32

static void BM_Cast_Compute_CF32_CF32(bm::State& state) {
    CastTest<Modules::Cast, CF32, CF32> mud;
    BL_CHECK_THROW(mud.run(state));
}

BENCHMARK(BM_Cast_Compute_CF32_CF32)
    ->Iterations(2<<13)
    ->Args({2})
    ->Args({16})
    ->Args({64})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

// CI8 to CF32

static void BM_Cast_Compute_CI8_CF32(bm::State& state) {
    CastTest<Modules::Cast, CI8, CF32> mud;
    BL_CHECK_THROW(mud.run(state));
}

BENCHMARK(BM_Cast_Compute_CI8_CF32)
    ->Iterations(2<<13)
    ->Args({2})
    ->Args({16})
    ->Args({64})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

// CI8 to CF16

static void BM_Cast_Compute_CI8_CF16(bm::State& state) {
    CastTest<Modules::Cast, CI8, CF16> mud;
    BL_CHECK_THROW(mud.run(state));
}

BENCHMARK(BM_Cast_Compute_CI8_CF16)
    ->Iterations(2<<13)
    ->Args({2})
    ->Args({16})
    ->Args({64})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);