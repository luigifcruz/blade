#include "./base.hh"

#include "blade/modules/detector.hh"

using namespace Blade;
namespace bm = benchmark;

static void BM_Detector_Compute(bm::State& state) {
    DetectorTest<Modules::Detector, CF32, F32> mud;
    BL_CHECK_THROW(mud.run(state));
}

BENCHMARK(BM_Detector_Compute)
    ->Iterations(2<<13)
    ->Args({2,  8, 4})
    ->Args({16, 8, 4})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_Detector_Compute)
    ->Iterations(2<<13)
    ->Args({2,  32, 4})
    ->Args({16, 32, 4})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_Detector_Compute)
    ->Iterations(2<<13)
    ->Args({2,  64, 4})
    ->Args({16, 64, 4})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_Detector_Compute)
    ->Iterations(2<<13)
    ->Args({2,  64, 1})
    ->Args({16, 64, 1})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);