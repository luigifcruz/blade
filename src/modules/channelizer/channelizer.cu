#include <cuComplex.h>
#include <cuda_fp16.h>
#include <stdint.h>

template<uint64_t N>
__global__ void shuffle(const cuFloatComplex* input, const uint64_t* indices, cuFloatComplex* output) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N){
        output[tid] = input[indices[tid]];
    }
}
