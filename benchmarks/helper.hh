#ifndef BLADE_BENCHMARK_HELPER_HH
#define BLADE_BENCHMARK_HELPER_HH

#include <cuda_runtime.h>
#include <benchmark/benchmark.h>

#include "blade/types.hh"
#include "blade/logger.hh"

namespace Blade {

class CudaBenchmark {
 protected:
    CudaBenchmark() {
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    }

    ~CudaBenchmark() {
        cudaStreamDestroy(stream);
    }

    const Result startIteration() {
        cudaEventCreate(&start);
        cudaEventRecord(start, stream);

        return Result::SUCCESS;
    }

    const Result finishIteration(benchmark::State& state) {
        cudaEventCreate(&stop);
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&elapsedTime, start, stop);
        state.SetIterationTime(elapsedTime / 1000);

        return Result::SUCCESS;
    }

    cudaStream_t& getStream() {
        return stream;
    }

    template<typename Block>
    static void Create(std::shared_ptr<Block>& module,
                       const typename Block::Config& config,
                       const typename Block::Input& input, 
                       const cudaStream_t& stream) {
        module = std::make_unique<Block>(config, input, stream);
    }

 private:
    cudaEvent_t start, stop;
    cudaStream_t stream;
    float elapsedTime;
};

}  // namespace Blade

#endif 
