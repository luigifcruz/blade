#include <cuComplex.h>

// 4-point FFT

__device__ __inline__ void fft_4pnt_proc(const char2 a, const char2 b, const char2 c, const char2 d,
                              char2* A, char2* B, char2* C, char2* D)
{
const char r1 = a.x - c.x;
    const char r2 = a.y - c.y;
    const char r3 = b.x - d.x;
    const char r4 = b.y - d.y;

    const char t1 = a.x + c.x;
    const char t2 = a.y + c.y;
    const char t3 = b.x + d.x;
    const char t4 = b.y + d.y;

    const char a3 = t1 - t3;
    const char a4 = t2 - t4;
    const char b3 = r1 - r4;
    const char b2 = r2 - r3;

    const char a1 = t1 + t3;
    const char a2 = t2 + t4;
    const char b1 = r1 + r4;
    const char b4 = r2 + r3;

    *A = make_char2(a1, a2);
    *B = make_char2(b1, b2);
    *C = make_char2(a3, a4);
    *D = make_char2(b3, b4);
}

template<size_t N, size_t FFT_SIZE, size_t NPOLS>
__global__ void fft_4pnt(const char2* input, char2* output) {
    const int numThreads = (blockDim.x * gridDim.x) * (FFT_SIZE * NPOLS);
    const int threadID = (blockIdx.x * blockDim.x + threadIdx.x) * (FFT_SIZE * NPOLS);

    for (int i = threadID; i < N; i += numThreads) {
        for (int j = i; j < i + NPOLS; j += 1) {
            fft_4pnt_proc(input[j+0], input[j+2], input[j+4], input[j+6],
                          &output[j+0], &output[j+2], &output[j+4], &output[j+6]);
        }
    }
}
