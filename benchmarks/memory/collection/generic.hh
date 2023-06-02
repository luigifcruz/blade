#ifndef BLADE_BENCHMARK_MEMORY_COLLECTION_GENERIC_HH
#define BLADE_BENCHMARK_MEMORY_COLLECTION_GENERIC_HH

#include "blade/memory/collection.hh"

#include "../../helper.hh"

namespace Blade {

template<typename T>
class CollectionTest : CudaBenchmark {
 public:
    Result runAccumulateBenchmark(benchmark::State& state) {
        ArrayTensor<Device::CUDA, T> input({27, 131072, 1, 2});
        ArrayTensor<Device::CUDA, T> output({27, 131072, 64, 2});

        // Run once for JIT cache to be filled.
        Memory::Collection::Accumulate(output, input, 2, 0, this->getStream());

        for (auto _ : state) {
            BL_CHECK(this->startIteration());
            
            {
                Memory::Collection::Accumulate(output, input, 2, 0, this->getStream());
            }

            BL_CHECK(this->finishIteration(state));
        }

        return Result::SUCCESS;
    }

    Result runAccumulateMemcpyBenchmark(benchmark::State& state) {
        ArrayTensor<Device::CUDA, T> input({27, 131072, 1, 2});
        ArrayTensor<Device::CUDA, T> output({27, 131072, 64, 2});

        for (auto _ : state) {
            BL_CHECK(this->startIteration());
            
            {
                // Accumulate AFTP buffers across the T dimension
                const auto& inputHeight = output.shape().numberOfAspects() * 
                                          output.shape().numberOfFrequencyChannels();
                const auto& inputWidth = input.size_bytes() / inputHeight;

                const auto& outputPitch = inputWidth * 64;

                BL_CHECK(
                    Memory::Copy2D(
                        output,
                        outputPitch, // dstStride
                        0 * inputWidth, // dstOffset

                        input,
                        inputWidth,
                        0,

                        inputWidth,
                        inputHeight, 
                        this->getStream()
                    )
                );
            }

            BL_CHECK(this->finishIteration(state));
        }

        return Result::SUCCESS;
    }

    Result runAccumulateSOLBenchmark(benchmark::State& state) {
        ArrayTensor<Device::CUDA, T> input({27, 131072, 1, 2});
        ArrayTensor<Device::CUDA, T> output({27, 131072, 1, 2});

        for (auto _ : state) {
            BL_CHECK(this->startIteration());
            
            {
                BL_CHECK(Memory::Copy(output, input, this->getStream()));
            }

            BL_CHECK(this->finishIteration(state));
        }

        return Result::SUCCESS;
    }
};



}  // namespace Blade

#endif
