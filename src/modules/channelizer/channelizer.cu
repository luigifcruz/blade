#include <cuComplex.h>

// 4-point FFT

// TODO: Add multiple formats support.
template<size_t N, size_t NFFT, size_t NPOLS>
__global__ void fft_4pnt(const cuFloatComplex* input, cuFloatComplex* output) {
    const int numThreads = (blockDim.x * gridDim.x) * (NFFT * NPOLS);
    const int threadID = (blockIdx.x * blockDim.x + threadIdx.x) * (NFFT * NPOLS);

    const int pol_index[] = {
        NPOLS * 0,
        NPOLS * 1,
        NPOLS * 2,
        NPOLS * 3,
    };

    for (int i = threadID; i < N; i += numThreads) {
        for (int j = i; j < i + NPOLS; j += 1) {
            // TODO: Add reordering index.

            const float2 a = input[j + pol_index[0]];
            const float2 b = input[j + pol_index[1]];
            const float2 c = input[j + pol_index[2]];
            const float2 d = input[j + pol_index[3]];

            const float r1 = a.x - c.x;
            const float r2 = a.y - c.y;
            const float r3 = b.x - d.x;
            const float r4 = b.y - d.y;

            const float t1 = a.x + c.x;
            const float t2 = a.y + c.y;
            const float t3 = b.x + d.x;
            const float t4 = b.y + d.y;

            const float a3 = t1 - t3;
            const float a4 = t2 - t4;
            const float b3 = r1 - r4;
            const float b2 = r2 - r3;

            const float a1 = t1 + t3;
            const float a2 = t2 + t4;
            const float b1 = r1 + r4;
            const float b4 = r2 + r3;

            output[j + pol_index[0]] = make_cuFloatComplex(a1, a2);
            output[j + pol_index[1]] = make_cuFloatComplex(b1, b2);
            output[j + pol_index[2]] = make_cuFloatComplex(a3, a4);
            output[j + pol_index[3]] = make_cuFloatComplex(b3, b4);
        }
    }
}
