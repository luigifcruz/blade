#include <chrono>

#include "blade/modules/beamformer/ata.hh"

#include <cuda_runtime.h>
#include <benchmark/benchmark.h>

using namespace Blade;
namespace bm = benchmark;

using IT = CF32;
using OT = CF32;
using MUT = Modules::Beamformer::ATA<IT, OT>;

static void BM_BeamformerATA_Processing(bm::State& state) {
    cudaEvent_t start, stop;
    cudaStream_t stream;
    float elapsedTime;

    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    const MUT::Config config {
        .enableIncoherentBeam = false,
        .enableIncoherentBeamSqrt = false,
        .blockSize = 512,
    };

    ArrayTensor<Device::CUDA, IT> inputBuf({
        .A = static_cast<const U64>(state.range(1)),
        .F = 192,
        .T = 8192,
        .P = 2,
    });
    PhasorTensor<Device::CUDA, IT> inputPhasors({
        .B = static_cast<const U64>(state.range(0)),
        .A = static_cast<const U64>(state.range(1)),
        .F = 192,
        .T = 1,
        .P = 2,
    });

    MUT module(config, {
        inputBuf, 
        inputPhasors,
    });
    
    for (auto _ : state) {
        cudaEventCreate(&start);
        cudaEventRecord(start, stream);
        
        {
            BL_CHECK_THROW(module.preprocess(stream, 0));
            BL_CHECK_THROW(module.process(stream));
        }

        cudaEventCreate(&stop);
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&elapsedTime, start, stop);
        state.SetIterationTime(elapsedTime / 1000);
    }

    cudaStreamDestroy(stream);
}

BENCHMARK(BM_BeamformerATA_Processing)
    ->Iterations(2<<13)
    ->Args({1, 20})
    ->Args({2, 20})
    ->Args({8, 20})
    ->Args({2, 42})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

static void BM_BeamformerATA_Transfer(bm::State& state) {
    cudaEvent_t start, stop;
    cudaStream_t stream;
    float elapsedTime;

    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    ArrayTensor<Device::CUDA, IT> inputBuf({
        .A = static_cast<const U64>(state.range(0)),
        .F = 192,
        .T = 8192,
        .P = 2,
    });
    PhasorTensor<Device::CUDA, IT> inputPhasors({
        .B = 8,
        .A = static_cast<const U64>(state.range(0)),
        .F = 192,
        .T = 1,
        .P = 2,
    });

    ArrayTensor<Device::CPU, IT> inputBufHost({
        .A = static_cast<const U64>(state.range(0)),
        .F = 192,
        .T = 8192,
        .P = 2,
    });
    PhasorTensor<Device::CPU, IT> inputPhasorsHost({
        .B = 8,
        .A = static_cast<const U64>(state.range(0)),
        .F = 192,
        .T = 1,
        .P = 2,
    });
    
    for (auto _ : state) {
        cudaEventCreate(&start);
        cudaEventRecord(start, stream);
        
        {
            BL_CHECK_THROW(Memory::Copy(inputBuf, inputBufHost, stream));
            BL_CHECK_THROW(Memory::Copy(inputPhasors, inputPhasorsHost, stream));
        }

        cudaEventCreate(&stop);
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&elapsedTime, start, stop);
        state.SetIterationTime(elapsedTime / 1000);
    }

    cudaStreamDestroy(stream);
}

BENCHMARK(BM_BeamformerATA_Transfer)
    ->Iterations(64)
    ->Args({20})
    ->Args({42})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK_MAIN();
