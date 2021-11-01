#include <cuda_fp16.h>

template<typename T, size_t N>
__global__ void checker(T a, T b, unsigned long long int* result) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N){
        if (abs(static_cast<double>(a[tid]) - static_cast<double>(b[tid])) > 0.1) {
            atomicAdd(result, 1);
        }
    }
}
