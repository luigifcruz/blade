#include <chrono>

#include "./mode_b.hh"
#include "../../../helper.hh"

using namespace Blade;
namespace bm = benchmark;

static void BM_PipelineModeB(benchmark::State& state) {
    const uint64_t count = 128;
    std::shared_ptr<Benchmark<CI8, CF32>> bench;

    BL_DISABLE_PRINT();
    Blade::InitAndProfile([&](){
        bench = std::make_shared<Benchmark<CI8, CF32>>();
    }, state);
    BL_ENABLE_PRINT();

    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();

        BL_DISABLE_PRINT();
        if (bench->run(count) != Result::SUCCESS) {
            BL_CHECK_THROW(Result::ERROR);
        }
        BL_ENABLE_PRINT();

        auto end = std::chrono::high_resolution_clock::now();

        auto elapsed_seconds =
          std::chrono::duration_cast<std::chrono::duration<double>>(
            end - start);

        state.SetIterationTime(elapsed_seconds.count() / count);
    }

    BL_DISABLE_PRINT();
    bench.reset();
    BL_ENABLE_PRINT();
}

BENCHMARK(BM_PipelineModeB)
    ->Iterations(1)
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK_MAIN();
