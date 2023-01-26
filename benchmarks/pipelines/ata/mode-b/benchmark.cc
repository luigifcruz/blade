#include <chrono>

#include <benchmark/benchmark.h>
#include <blade/logger.hh>
#include <blade/types.hh>

#define CHECK_THROW(a) if (a != 0) throw "Pipeline error.";

namespace bm = benchmark;

extern "C" {
#include "mode_b_stub.h"
}

static void BM_PipelineModeB(benchmark::State& state) {
    const uint64_t count = 2048;

    BL_DISABLE_PRINT();
    CHECK_THROW(mode_b_setup());
    BL_ENABLE_PRINT();

    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();

        BL_DISABLE_PRINT();
        CHECK_THROW(mode_b_loop(count));
        BL_ENABLE_PRINT();

        auto end = std::chrono::high_resolution_clock::now();

        auto elapsed_seconds =
          std::chrono::duration_cast<std::chrono::duration<double>>(
            end - start);

        state.SetIterationTime(elapsed_seconds.count() / count);
    }

    BL_DISABLE_PRINT();
    CHECK_THROW(mode_b_terminate());
    BL_ENABLE_PRINT();
}

BENCHMARK(BM_PipelineModeB)
    ->Iterations(8)
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK_MAIN();
