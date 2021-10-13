#include <cuComplex.h>

template<size_t N>
__global__ void FOUR_PNT_FFT(char2* input) {
    const int numThreads = blockDim.x * gridDim.x * 4;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = threadID; i < N; i += numThreads) {
    }
}
