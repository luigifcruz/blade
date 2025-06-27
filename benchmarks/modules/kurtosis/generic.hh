#include "./base.hh"

using namespace Blade;
namespace bm = benchmark;

// CF32 -> CF32


static void BM_Kurtosis_Compute_CF32_CF32(bm::State& state) {
    KurtosisTest<Modules::Kurtosis, CF32, CF32> mud;
    BL_CHECK_THROW(mud.run(state));
}

BENCHMARK(BM_Kurtosis_Compute_CF32_CF32)
    ->Iterations(2<<8)
    //->Args({ 2, static_cast<uint8_t>(POL::LR)})
    //->Args({16, static_cast<uint8_t>(POL::LR)})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

// CF16 -> CF16
/*
static void BM_Kurtosis_Compute_CF16_CF16(bm::State& state) {
    KurtosisTest<Modules::Kurtosis, CF16, CF16> mud;
    BL_CHECK_THROW(mud.run(state));
}

BENCHMARK(BM_Kurtosis_Compute_CF16_CF16)
    ->Iterations(2<<13)
    //->Args({ 2, static_cast<uint8_t>(POL::LR)})
    //->Args({16, static_cast<uint8_t>(POL::LR)})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

*/
