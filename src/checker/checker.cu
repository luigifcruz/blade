#include <cuda_fp16.h>

template<typename T, size_t N, size_t S>
__global__ void checker(T a, T b, unsigned long long int* result) {
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = threadID; i < N; i += numThreads) {
        if (abs(static_cast<double>(a[i]) - static_cast<double>(b[i])) > 0.1) {
            atomicAdd(result, 1);
        }
    }

    if (threadID == 0) {
        *result /= S;
    }
}
