#include "./base.hh"

using namespace Blade;
namespace bm = benchmark;

static void BM_Permutation_Compute(bm::State& state) {
    PermutationTest<Modules::Permutation, CF32, CF32> mud;
    BL_CHECK_THROW(mud.run(state));
}

BENCHMARK(BM_Permutation_Compute)
    ->Iterations(2<<13)
    ->Args({0, 1, 2, 3})  // AFTP (Identity)
    ->Args({0, 2, 1, 3})  // ATFP
    ->Args({0, 2, 3, 1})  // ATPF
    ->UseManualTime()
    ->Unit(bm::kMillisecond);