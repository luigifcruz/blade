#include <cuComplex.h>

// 4-point FFT

// TODO: Add multiple formats support.
template<size_t N, size_t NFFT, size_t NPOLS>
__global__ void fft_4pnt(const cuFloatComplex* input, cuFloatComplex* output) {
    const int numThreads = (blockDim.x * gridDim.x) * (NFFT * NPOLS);
    const int threadID = (blockIdx.x * blockDim.x + threadIdx.x) * (NFFT * NPOLS);


}
