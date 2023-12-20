#include <cuComplex.h>
#include <cuda_fp16.h>
#include <stdint.h>

template<uint64_t P, uint64_t F, uint64_t N>
__global__ void pre_channelizer(const cuFloatComplex* input, cuFloatComplex* output) {
    const uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N){
        auto element = input[tid];

        if ((tid % P) >= F){
            element.x = -element.x;
            element.y = -element.y;
        }

        output[tid] = element;
    }
}