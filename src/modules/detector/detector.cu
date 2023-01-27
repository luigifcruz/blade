#include <cuda_fp16.h>
#include <stdint.h>
#include "cuComplex.h"

// Number of input polarizations is always 2, and output polarizations is static per kernel

template<uint64_t A, uint64_t F, uint64_t T, uint64_t INTG>
__global__ void detector_4pol_AFTP(const cuFloatComplex* input,
                              float* output,
                              const bool* resetTensor) {
    const uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if (*resetTensor) {
        if (index < (A*F*T / INTG)) {
            reinterpret_cast<float4*>(output)[index] = {0.0, 0.0, 0.0, 0.0};
        }
        __syncthreads();
    }

    if (index < A*F*T) {
        const float4 sample = reinterpret_cast<const float4*>(input)[index];
        
        cuFloatComplex sample_X = make_cuFloatComplex(sample.x, sample.y);
        cuFloatComplex sample_Y = make_cuFloatComplex(sample.z, sample.w);

        cuFloatComplex X = cuCmulf(sample_X, cuConjf(sample_X));
        cuFloatComplex Y = cuCmulf(sample_Y, cuConjf(sample_Y));
        cuFloatComplex Z = cuCmulf(sample_X, cuConjf(sample_Y));

        const uint64_t oid = (index / INTG) * 4;
        atomicAdd(output + oid + 0, X.x);
        atomicAdd(output + oid + 1, Y.x);
        atomicAdd(output + oid + 2, Z.x);
        atomicAdd(output + oid + 3, Z.y);
    }
}

template<uint64_t A, uint64_t F, uint64_t T, uint64_t INTG>
__global__ void detector_1pol_AFTP(const cuFloatComplex* input,
                              float* output,
                              const bool* resetTensor) {
    const uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if (*resetTensor) {
        if (index < (A*F*T / INTG)) {
            output[index] = 0.0;
        }
        __syncthreads();
    }

    if (index < A*F*T) {
        const float4 sample = reinterpret_cast<const float4*>(input)[index];

        const float X = sample.x * sample.x + sample.y * sample.y;
        const float Y = sample.z * sample.z + sample.w * sample.w;

        atomicAdd(output + (index / INTG), X + Y);
    }
}

template<uint64_t A, uint64_t F, uint64_t T, uint64_t INTG>
__global__ void detector_4pol_ATPF(const cuFloatComplex* input,
                              float* output,
                              const bool* resetTensor) {
    const uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    // AFTP -> ATPF
    const uint64_t t = index % T;
    const uint64_t f = (index / T) % F;
    const uint64_t a = index / (F*T);
    const uint64_t index_out = ((a*T + t)/INTG*4 + 0)*F + f;

    if (*resetTensor) {
        if (index < (A*F*T / INTG)) {
            reinterpret_cast<float4*>(output)[index] = {0.0, 0.0, 0.0, 0.0};
        }
        __syncthreads();
    }

    if (index < A*F*T) {
        const float4 sample = reinterpret_cast<const float4*>(input)[index];
        
        cuFloatComplex sample_X = make_cuFloatComplex(sample.x, sample.y);
        cuFloatComplex sample_Y = make_cuFloatComplex(sample.z, sample.w);

        cuFloatComplex X = cuCmulf(sample_X, cuConjf(sample_X));
        cuFloatComplex Y = cuCmulf(sample_Y, cuConjf(sample_Y));
        cuFloatComplex Z = cuCmulf(sample_X, cuConjf(sample_Y));

        atomicAdd(output + index_out + 0*F, X.x);
        atomicAdd(output + index_out + 1*F, Y.x);
        atomicAdd(output + index_out + 2*F, Z.x);
        atomicAdd(output + index_out + 3*F, Z.y);
    }
}

template<uint64_t A, uint64_t F, uint64_t T, uint64_t INTG>
__global__ void detector_1pol_ATPF(const cuFloatComplex* input,
                              float* output,
                              const bool* resetTensor) {
    const uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    // AFTP -> ATPF
    const uint64_t t = index % T;
    const uint64_t f = (index / T) % F;
    const uint64_t a = index / (F*T);
    const uint64_t index_out = ((a*T + t)/INTG*1 + 0)*F + f;

    if (*resetTensor) {
        if (index < (A*F*T / INTG)) {
            output[index_out] = 0.0;
        }
        __syncthreads();
    }

    if (index < A*F*T) {
        const float4 sample = reinterpret_cast<const float4*>(input)[index];

        const float X = sample.x * sample.x + sample.y * sample.y;
        const float Y = sample.z * sample.z + sample.w * sample.w;

        atomicAdd(output + index_out, X + Y);
    }
}

template<uint64_t A, uint64_t F, uint64_t T, uint64_t INTG>
__global__ void detector_4pol_ATPFrev(const cuFloatComplex* input,
                              float* output,
                              const bool* resetTensor) {
    const uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    // AFTP -> ATPFrev
    const uint64_t t = index % T;
    const uint64_t f = (index / T) % F;
    const uint64_t a = index / (F*T);
    const uint64_t index_out = ((a*T + t)/INTG*4 + 0)*F + F-1-f;

    if (*resetTensor) {
        if (index < (A*F*T / INTG)) {
            reinterpret_cast<float4*>(output)[index] = {0.0, 0.0, 0.0, 0.0};
        }
        __syncthreads();
    }

    if (index < A*F*T) {
        const float4 sample = reinterpret_cast<const float4*>(input)[index];
        
        cuFloatComplex sample_X = make_cuFloatComplex(sample.x, sample.y);
        cuFloatComplex sample_Y = make_cuFloatComplex(sample.z, sample.w);

        cuFloatComplex X = cuCmulf(sample_X, cuConjf(sample_X));
        cuFloatComplex Y = cuCmulf(sample_Y, cuConjf(sample_Y));
        cuFloatComplex Z = cuCmulf(sample_X, cuConjf(sample_Y));

        atomicAdd(output + index_out + 0*F, X.x);
        atomicAdd(output + index_out + 1*F, Y.x);
        atomicAdd(output + index_out + 2*F, Z.x);
        atomicAdd(output + index_out + 3*F, Z.y);
    }
}

template<uint64_t A, uint64_t F, uint64_t T, uint64_t INTG>
__global__ void detector_1pol_ATPFrev(const cuFloatComplex* input,
                              float* output,
                              const bool* resetTensor) {
    const uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    // AFTP -> ATPFrev
    const uint64_t t = index % T;
    const uint64_t f = (index / T) % F;
    const uint64_t a = index / (F*T);
    const uint64_t index_out = ((a*T + t)/INTG*1 + 0)*F + F-1-f;
    // const uint64_t index_out = ((a*F + F-1-f)*T + t)/INTG;

    if (*resetTensor) {
        if (index < (A*F*T / INTG)) {
            output[index_out] = 0.0;
        }
        __syncthreads();
    }

    if (index < A*F*T) {
        const float4 sample = reinterpret_cast<const float4*>(input)[index];

        const float X = sample.x * sample.x + sample.y * sample.y;
        const float Y = sample.z * sample.z + sample.w * sample.w;

        atomicAdd(output + index_out, X + Y);
    }
}