#include <cuComplex.h>
#include <cuda_fp16.h>
#include <stdint.h>

template<typename IT, typename OT, uint64_t N>
__global__ void polarizer(const IT* input, OT* output) {
    const int tid = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

    if (tid < (N * 2)) {
        const IT& xPol = input[tid + 0];
        const OT& yPol = input[tid + 1];
        output[tid + 0] = xPol + yPol;
        output[tid + 1] = xPol - yPol;
    }
}
