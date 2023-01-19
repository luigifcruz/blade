#include <chrono>

#include <benchmark/benchmark.h>

namespace bm = benchmark;

static void BM_BeamformerATA(bm::State& state) {
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Simulate some useful workload with a sleep

        auto end = std::chrono::high_resolution_clock::now();

        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(
            end - start);

        state.SetIterationTime(elapsed_seconds.count());
    }
}

BENCHMARK(BM_BeamformerATA)->Range(1, 1<<17)->UseManualTime();

BENCHMARK_MAIN();
