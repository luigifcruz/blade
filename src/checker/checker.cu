template<typename T, size_t N>
__global__ void checker(T a, T b, unsigned long long int* result) {
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = threadID; i < N; i += numThreads) {
        if (abs((double)a[i] - (double)b[i]) > 0.1) {
            atomicAdd(result, 1);
        }
    }
}
