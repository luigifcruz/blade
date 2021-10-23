#include <cuda_fp16.h>

template<typename IT, typename OT, size_t N>
__global__ void cast(IT* input, OT* output) {
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = threadID; i < N; i += numThreads) {
        output[i] = static_cast<OT>(input[i]);
    }
}
