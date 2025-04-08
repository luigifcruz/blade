#include "./base.hh"

#include "blade/modules/correlator.hh"

using namespace Blade;
namespace bm = benchmark;

static void BM_Correlator_Compute_CF32(bm::State& state) {
    CorrelatorTest<Modules::Correlator, CF32, CF32> mud;
    BL_CHECK_THROW(mud.run(state));
}

static void BM_Correlator_Compute_CI8(bm::State& state) {
    CorrelatorTest<Modules::Correlator, CI8, CF32> mud;
    BL_CHECK_THROW(mud.run(state));
}

// ATA Standard Mode

BENCHMARK(BM_Correlator_Compute_CF32)
    ->Iterations(2<<8)
    ->Args({28, 192, 8192, 2, 1, 0, 0, 64})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_Correlator_Compute_CI8)
    ->Iterations(2<<8)
    ->Args({28, 192, 8192, 2, 1, 0, 0, 64})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_Correlator_Compute_CF32)
    ->Iterations(2<<8)
    ->Args({28, 192, 8192, 2, 1, 0, 1, 64})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_Correlator_Compute_CI8)
    ->Iterations(2<<8)
    ->Args({28, 192, 8192, 2, 1, 0, 1, 64})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_Correlator_Compute_CF32)
    ->Iterations(2<<8)
    ->Args({28, 192, 8192, 2, 1, 0, 2, 64})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_Correlator_Compute_CI8)
    ->Iterations(2<<8)
    ->Args({28, 192, 8192, 2, 1, 0, 2, 64})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_Correlator_Compute_CI8)
    ->Iterations(2<<8)
    ->Args({28, 192, 8192, 2, 1, 1, 0, 64})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_Correlator_Compute_CI8)
    ->Iterations(2<<8)
    ->Args({28, 192, 8192, 2, 1, 1, 1, 64})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_Correlator_Compute_CI8)
    ->Iterations(2<<8)
    ->Args({28, 192, 8192, 2, 1, 1, 2, 64})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

// ATA High-Resolution Mode

BENCHMARK(BM_Correlator_Compute_CF32)
    ->Iterations(2<<8)
    ->Args({28, 65536, 1, 2, 1, 0, 0, 32})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_Correlator_Compute_CI8)
    ->Iterations(2<<8)
    ->Args({28, 65536, 1, 2, 1, 0, 0, 32})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_Correlator_Compute_CF32)
    ->Iterations(2<<8)
    ->Args({28, 65536, 1, 2, 1, 0, 1, 32})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_Correlator_Compute_CI8)
    ->Iterations(2<<8)
    ->Args({28, 65536, 1, 2, 1, 0, 1, 32})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_Correlator_Compute_CF32)
    ->Iterations(2<<8)
    ->Args({28, 65536, 1, 2, 1, 0, 2, 32})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_Correlator_Compute_CI8)
    ->Iterations(2<<8)
    ->Args({28, 65536, 1, 2, 1, 0, 2, 32})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_Correlator_Compute_CI8)
    ->Iterations(2<<8)
    ->Args({28, 65536, 1, 2, 1, 1, 0, 32})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_Correlator_Compute_CI8)
    ->Iterations(2<<8)
    ->Args({28, 65536, 1, 2, 1, 1, 1, 32})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_Correlator_Compute_CI8)
    ->Iterations(2<<8)
    ->Args({28, 65536, 1, 2, 1, 1, 2, 32})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);
