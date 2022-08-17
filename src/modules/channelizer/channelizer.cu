#include <cuComplex.h>
#include <cuda_fp16.h>
#include <stdint.h>

template<uint64_t N, uint64_t NFFT, uint64_t NPOLS, uint64_t NTIME, uint64_t NCHANS>
__global__ void fft_4pnt(const cuFloatComplex* input, cuFloatComplex* output) {
    const int numThreads = (blockDim.x * gridDim.x) * (NFFT * NPOLS);
    const int threadID = (blockIdx.x * blockDim.x + threadIdx.x) * (NFFT * NPOLS);

    const int pol_index[] = {
        NPOLS * 0,
        NPOLS * 1,
        NPOLS * 2,
        NPOLS * 3,
    };

    const int ch_pitch = NTIME / NFFT;
    const int ch_index[] = {
        ch_pitch * pol_index[0],
        ch_pitch * pol_index[1],
        ch_pitch * pol_index[2],
        ch_pitch * pol_index[3],
    };

    const int ant_pitch = NCHANS * NTIME * NPOLS;
    const int och_pitch = NTIME * NPOLS;

    for (int i = threadID; i < N; i += numThreads) {
        const int ant_id = i / ant_pitch;
        const int ant_offset = ant_id * ant_pitch;

        const int ch_id = (i - ant_offset) / och_pitch;
        const int ch_offset = ch_id * och_pitch;

        const int io = ((i - ant_offset - ch_offset) / NFFT) + ant_offset + ch_offset;

        for (int j = 0; j < NPOLS; j += 1) {
#ifdef KERN_DEBUG
            printf("%d - %d %d %d %d - %d %d %d %d | %d %d %d | %d %d\n",
                i,

                j+i+pol_index[0],
                j+i+pol_index[1],
                j+i+pol_index[2],
                j+i+pol_index[3],

                j+io+ch_index[0],
                j+io+ch_index[1],
                j+io+ch_index[2],
                j+io+ch_index[3],

                i, ch_pitch, sauce,
                iant, ichan);
#endif

            const float2 a = input[i + j + pol_index[0]];
            const float2 b = input[i + j + pol_index[1]];
            const float2 c = input[i + j + pol_index[2]];
            const float2 d = input[i + j + pol_index[3]];

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

            output[io + j + ch_index[2]] = make_cuFloatComplex(a1, a2);
            output[io + j + ch_index[3]] = make_cuFloatComplex(b1, b2);
            output[io + j + ch_index[0]] = make_cuFloatComplex(a3, a4);
            output[io + j + ch_index[1]] = make_cuFloatComplex(b3, b4);
        }
    }
}

template<uint64_t N, uint64_t NPOLS>
__global__ void shifter(const cuFloatComplex* in, cuFloatComplex* out) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N) {
        cuFloatComplex tmp = in[tid];

        if (((tid / NPOLS) % 2) != 0){
            tmp.x = -tmp.x;
            tmp.y = -tmp.y;
        }

        out[tid] = tmp;
    }
}

template<uint64_t N>
__global__ void shuffler(const cuFloatComplex* input, const uint64_t* indices, cuFloatComplex* output) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N){
        output[tid] = input[indices[tid]];
    }
}
