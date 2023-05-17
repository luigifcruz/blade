#include <benchmark/benchmark.h>
#include <cuda_fp16.h>
#include <cuComplex.h>

#define BL_OPS_HOST_SIDE_KEY
#include "blade/memory/ops.hh"
#include "blade/memory/vector.hh"
#include "blade/memory/custom.hh"

using namespace Blade;

enum class ArithmeticOp : uint64_t {
    NOOP = 0,
    ADD  = 1,
    MULT = 2,
    DIV  = 3,
    SUB  = 4,
};

template <typename T, ArithmeticOp Op>
__global__ void OpsComplexKernel(const ops::complex<T>* a, const ops::complex<T>* b, ops::complex<T>* c, uint64_t n) {
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if constexpr (Op == ArithmeticOp::NOOP) {
        }
        if constexpr (Op == ArithmeticOp::ADD) {
            c[i] = a[i] + b[i];
        }
        if constexpr (Op == ArithmeticOp::SUB) {
            c[i] = a[i] - b[i];
        }
        if constexpr (Op == ArithmeticOp::MULT) {
            c[i] = a[i] * b[i];
        }
        if constexpr (Op == ArithmeticOp::DIV) {
            c[i] = a[i] / b[i];
        }
    }
}

template <ArithmeticOp Op>
__global__ void CuComplexKernel(const cuComplex* a, const cuComplex* b, cuComplex* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if constexpr (Op == ArithmeticOp::NOOP) {
        }
        if constexpr (Op == ArithmeticOp::ADD) {
            c[i] = cuCaddf(a[i], b[i]);
        }
        if constexpr (Op == ArithmeticOp::SUB) {
            c[i] = cuCsubf(a[i], b[i]);
        }
        if constexpr (Op == ArithmeticOp::MULT) {
            c[i] = cuCmulf(a[i], b[i]);
        }
        if constexpr (Op == ArithmeticOp::DIV) {
            c[i] = cuCdivf(a[i], b[i]);
        }
    }
}

template <typename T, ArithmeticOp Op>
static void OpsComplexKernelBenchmark(benchmark::State& state) {
    U64 n = state.range(0);
    U64 block_size = 256;
    U64 num_blocks = (n + block_size - 1) / block_size;

    Tensor<Device::CUDA, std::complex<T>> a({n}, true);
    Tensor<Device::CUDA, std::complex<T>> b({n}, true);
    Tensor<Device::CUDA, std::complex<T>> c({n}, true);

    for (uint64_t i = 0; i < n; i++) {
        a[i] = std::complex<T>(T(1), T(2));
        b[i] = std::complex<T>(T(3), T(4));
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (auto _ : state) {
        cudaEventRecord(start);
        OpsComplexKernel<T, Op><<<num_blocks, block_size>>>(
                reinterpret_cast<ops::complex<T>*>(a.data()),
                reinterpret_cast<ops::complex<T>*>(b.data()),
                reinterpret_cast<ops::complex<T>*>(c.data()), n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        state.SetIterationTime(elapsed_ms / 1000.0);
    }
}

template <ArithmeticOp Op>
static void CuComplexKernelBenchmark(benchmark::State& state) {
    U64 n = state.range(0);
    U64 block_size = 256;
    U64 num_blocks = (n + block_size - 1) / block_size;

    Tensor<Device::CUDA, std::complex<F32>> a({n}, true);
    Tensor<Device::CUDA, std::complex<F32>> b({n}, true);
    Tensor<Device::CUDA, std::complex<F32>> c({n}, true);

    for (uint64_t i = 0; i < n; i++) {
        a[i] = std::complex<F32>(F32(1), F32(2));
        b[i] = std::complex<F32>(F32(3), F32(4));
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (auto _ : state) {
        cudaEventRecord(start);
        CuComplexKernel<Op><<<num_blocks, block_size>>>(
                reinterpret_cast<cuComplex*>(a.data()),
                reinterpret_cast<cuComplex*>(b.data()),
                reinterpret_cast<cuComplex*>(c.data()), n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        state.SetIterationTime(elapsed_ms / 1000.0);
    }
}

BENCHMARK_TEMPLATE(OpsComplexKernelBenchmark, F16, ArithmeticOp::NOOP)
    ->RangeMultiplier(2)
    ->Range(2<<19, 2<<20)
    ->UseManualTime();

// ADD

BENCHMARK_TEMPLATE(OpsComplexKernelBenchmark, F16, ArithmeticOp::ADD)
    ->RangeMultiplier(2)
    ->Range(2<<19, 2<<20)
    ->UseManualTime();

BENCHMARK_TEMPLATE(OpsComplexKernelBenchmark, F32, ArithmeticOp::ADD)
    ->RangeMultiplier(2)
    ->Range(2<<19, 2<<20)
    ->UseManualTime();
    
BENCHMARK_TEMPLATE(OpsComplexKernelBenchmark, F64, ArithmeticOp::ADD)
    ->RangeMultiplier(2)
    ->Range(2<<19, 2<<20)
    ->UseManualTime();
    
BENCHMARK_TEMPLATE(CuComplexKernelBenchmark, ArithmeticOp::ADD)
    ->RangeMultiplier(2)
    ->Range(2<<19, 2<<20)
    ->UseManualTime();

// SUB

BENCHMARK_TEMPLATE(OpsComplexKernelBenchmark, F16, ArithmeticOp::SUB)
    ->RangeMultiplier(2)
    ->Range(2<<19, 2<<20)
    ->UseManualTime();

BENCHMARK_TEMPLATE(OpsComplexKernelBenchmark, F32, ArithmeticOp::SUB)
    ->RangeMultiplier(2)
    ->Range(2<<19, 2<<20)
    ->UseManualTime();
    
BENCHMARK_TEMPLATE(OpsComplexKernelBenchmark, F64, ArithmeticOp::SUB)
    ->RangeMultiplier(2)
    ->Range(2<<19, 2<<20)
    ->UseManualTime();
    
BENCHMARK_TEMPLATE(CuComplexKernelBenchmark, ArithmeticOp::SUB)
    ->RangeMultiplier(2)
    ->Range(2<<19, 2<<20)
    ->UseManualTime();

// MULT

BENCHMARK_TEMPLATE(OpsComplexKernelBenchmark, F16, ArithmeticOp::MULT)
    ->RangeMultiplier(2)
    ->Range(2<<19, 2<<20)
    ->UseManualTime();

BENCHMARK_TEMPLATE(OpsComplexKernelBenchmark, F32, ArithmeticOp::MULT)
    ->RangeMultiplier(2)
    ->Range(2<<19, 2<<20)
    ->UseManualTime();
    
BENCHMARK_TEMPLATE(OpsComplexKernelBenchmark, F64, ArithmeticOp::MULT)
    ->RangeMultiplier(2)
    ->Range(2<<19, 2<<20)
    ->UseManualTime();
    
BENCHMARK_TEMPLATE(CuComplexKernelBenchmark, ArithmeticOp::MULT)
    ->RangeMultiplier(2)
    ->Range(2<<19, 2<<20)
    ->UseManualTime();

// DIV

BENCHMARK_TEMPLATE(OpsComplexKernelBenchmark, F16, ArithmeticOp::DIV)
    ->RangeMultiplier(2)
    ->Range(2<<19, 2<<20)
    ->UseManualTime();

BENCHMARK_TEMPLATE(OpsComplexKernelBenchmark, F32, ArithmeticOp::DIV)
    ->RangeMultiplier(2)
    ->Range(2<<19, 2<<20)
    ->UseManualTime();
    
BENCHMARK_TEMPLATE(OpsComplexKernelBenchmark, F64, ArithmeticOp::DIV)
    ->RangeMultiplier(2)
    ->Range(2<<19, 2<<20)
    ->UseManualTime();
    
BENCHMARK_TEMPLATE(CuComplexKernelBenchmark, ArithmeticOp::DIV)
    ->RangeMultiplier(2)
    ->Range(2<<19, 2<<20)
    ->UseManualTime();

BENCHMARK_MAIN();
