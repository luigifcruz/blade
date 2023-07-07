#include <cuda_fp16.h>
#include <stdint.h>
#include "cuComplex.h"

// TODO: Convert to Ops.

template<uint64_t N, uint64_t INTG>
__global__ void detector_4pol(const cuFloatComplex* input,
                              float* output,
                              const bool* resetTensor) {
    const uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (*resetTensor) {
        if (tid < (N / INTG)) {
            reinterpret_cast<float4*>(output)[tid] = {0.0, 0.0, 0.0, 0.0};
        }
        __syncthreads();
    }

    if (tid < N) {
        const float4 sample = reinterpret_cast<const float4*>(input)[tid];
        
        const cuFloatComplex sample_X = make_cuFloatComplex(sample.x, sample.y);
        const cuFloatComplex sample_Y = make_cuFloatComplex(sample.z, sample.w);

        const cuFloatComplex X = cuCmulf(sample_X, cuConjf(sample_X));
        const cuFloatComplex Y = cuCmulf(sample_Y, cuConjf(sample_Y));
        const cuFloatComplex Z = cuCmulf(sample_X, cuConjf(sample_Y));

        const uint64_t oid = (tid / INTG) * 4;
        atomicAdd(output + oid + 0, X.x);
        atomicAdd(output + oid + 1, Y.x);
        atomicAdd(output + oid + 2, Z.x);
        atomicAdd(output + oid + 3, Z.y);
    }
}

template<uint64_t N, uint64_t INTG>
__global__ void detector_1pol(const cuFloatComplex* input,
                              float* output,
                              const bool* resetTensor) {
    const uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (*resetTensor) {
        if (tid < (N / INTG)) {
            output[tid] = 0.0;
        }
        __syncthreads();
    }

    if (tid < N) {
        const float4 sample = reinterpret_cast<const float4*>(input)[tid];

        const float X = sample.x * sample.x + sample.y * sample.y;
        const float Y = sample.z * sample.z + sample.w * sample.w;

        atomicAdd(output + (tid / INTG), X + Y);
    }
}
