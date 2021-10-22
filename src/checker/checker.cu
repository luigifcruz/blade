template<typename T, size_t N>
__global__ void checker(T a, T b, unsigned long long int* result) {
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = threadID; i < N; i += numThreads) {
#ifdef DEBUG
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            for (int i = 0; i < N; i++) {
                const char* status = (abs(a[i], b[i]) > 0.01) ? "NOK" : " OK";
                printf("[%04d] %s %f != %f\n", i, status, (T)a[i], (T)b[i]);
            }
        }
#endif
        if (abs((double)a[i] - (double)b[i]) > 0.01) {
            atomicAdd(result, 1);
        }
    }
}
