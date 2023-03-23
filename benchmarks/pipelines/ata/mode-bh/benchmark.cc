#include <chrono>

#include <benchmark/benchmark.h>
#include <blade/logger.hh>
#include <blade/types.hh>

#include "../../../helper.hh"

using namespace Blade;

#define CHECK_THROW(a) if (a != 0) throw "Pipeline error.";

namespace bm = benchmark;

extern "C" {
#include "mode_bh_stub.h"
}

static void BM_PipelineModeBH(benchmark::State& state) {
    const uint64_t count = 2048;

    BL_DISABLE_PRINT();
    BL_CHECK_THROW(Blade::InitAndProfile([&](){
        return (mode_bh_init()) ? Result::ERROR : Result::SUCCESS;
    }, state));
    CHECK_THROW(mode_bh_setup());
    BL_ENABLE_PRINT();

    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();

        BL_DISABLE_PRINT();
        CHECK_THROW(mode_bh_loop(count));
        BL_ENABLE_PRINT();

        auto end = std::chrono::high_resolution_clock::now();

        auto elapsed_seconds =
          std::chrono::duration_cast<std::chrono::duration<double>>(
            end - start);

        state.SetIterationTime(elapsed_seconds.count() / count);
    }

    BL_DISABLE_PRINT();
    CHECK_THROW(mode_bh_terminate());
    BL_ENABLE_PRINT();
}

BENCHMARK(BM_PipelineModeBH)
    ->Iterations(8)
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK_MAIN();
