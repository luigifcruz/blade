#include "blade/memory/base.hh"
#include "cuComplex.h"
#include <curand_kernel.h>

using namespace Blade;

// CUDA kernel to compute sk_array
template<typename IT, typename OT, bool debugMode,
    int N_ANTS, int N_CHANS, int N_SAMPS, int N_POLS,
    int nKurtosisSigma, int block_size>
__global__ void compute_sk_array(cuFloatComplex* block, U8* mask, int maskcounter) {
    // let's assume STDDEV of 5 for now

    constexpr int masksize = N_ANTS * N_CHANS * N_POLS * N_SAMPS / (block_size * 8);
    constexpr double quotient = (1.0 * block_size + 1) / (1.0 * block_size - 1);

    float sklim_lower, sklim_upper;

    switch (nKurtosisSigma) {
        case 3:
            sklim_lower = 0.698159;
            sklim_upper = 1.49597;
            break;
        case 4:
            sklim_lower = 0.613738;
            sklim_upper = 1.784;
            break;
        default: // default is 5
            sklim_lower = 0.526881;
            sklim_upper = 2.18694;
            break;
    }

    // Compute indices
    int ant = blockIdx.x;
    int chan = threadIdx.x + blockIdx.y * blockDim.x;
    if (chan >= N_CHANS) {
        return;
    }

    float v2_1, v2_2, sk_1, sk_2;
    float s1_1, s2_1, s1_2, s2_2;
    float x1, y1, x2, y2;

    int idx1, baseidx, intermediate_base;

    int threadbaseidx = (ant * N_CHANS + chan) * N_SAMPS;

    // float skvals[64];
    
    // for this ant-chan: one sk_val for each chan and pol
    float skvals[N_POLS * N_SAMPS / block_size];
    int skind = 0;
    for (int samp_start = 0; samp_start < N_SAMPS; samp_start = samp_start + block_size) {
        // set accumulators to 0
        s1_1 = 0;
        s2_1 = 0;

        s1_2 = 0;
        s2_2 = 0;

        baseidx = threadbaseidx + samp_start;
        intermediate_base = baseidx * N_POLS;
        idx1 = intermediate_base;

        for (int samp = 0; samp < block_size; samp++) {
            // load the two complex numbers into memory via a
            // four-float memory read
            asm volatile(
                "ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
                : "=f"(x1), "=f"(y1), "=f"(x2), "=f"(y2)
                : "l"(block + idx1)
                );

            // square all four and then sum into the v2_X values
            // to compute squared magnitudes
            x1 = x1 * x1;
            y1 = y1 * y1;
            x2 = x2 * x2;
            y2 = y2 * y2;

            v2_1 = x1 + y1;
            v2_2 = x2 + y2;
            
            // sum of squared magnitudes
            s1_1 += v2_1;
            s1_2 += v2_2;

            // sum of fourth-pow magnitudes
            // fmaf(a, b, c) is equivalent to c = c + a * b
            s2_1 = fmaf(v2_1, v2_1, s2_1);
            s2_2 = fmaf(v2_2, v2_2, s2_2);

            // jump ahead by N_POLS for next values
            idx1 = idx1 + N_POLS;
        }

        // Compute sk value
        sk_1 = quotient * ((block_size * (s2_1 / (s1_1 * s1_1))) - 1.0f);
        sk_2 = quotient * ((block_size * (s2_2 / (s1_2 * s1_2))) - 1.0f);

        // fill up the array
        skvals[skind++] = sk_1;
        skvals[skind++] = sk_2;
    }


    int chan_start;
    int zap1, zap2;
    skind = 0;
    int maskidx_raw;
    int maskidx_true;

    float replx1, reply1, replx2, reply2;
    double rmean;
    double imean;
    double rstd;
    double istd;

    // modify distributions based on if we are debugging
    if constexpr (debugMode) {
        rmean = 100.0f;
        imean= 100.0f;
        rstd = 0.0f;
        istd = 0.0f;
    } else {
        rmean = 0.0f;
        imean = 0.0f;
        rstd = 8.0f;
        istd = 8.0f;
    }


    curandState state;
    // ant * chan + ant just creates a unique id for each thread
    curand_init(1234ULL, ant * chan + ant, 0, &state);

    for (int kurtblock_idx = 0; kurtblock_idx < N_SAMPS/block_size; kurtblock_idx++) {
        intermediate_base = (threadbaseidx + kurtblock_idx*block_size) * N_POLS;

        // based on sk we can zap the channel
        sk_1 = skvals[skind++];
        sk_2 = skvals[skind++];
        
        zap1 = sk_1 < sklim_lower || sk_1 > sklim_upper;
        zap2 = sk_2 < sklim_lower || sk_2 > sklim_upper;

        // maskidx_raw corresponds to the element in a mask with no int-to-bit
        // reduction
        // maskidx_true corresponds to the element in a mask where 8 ints are
        //   reduced into one -- one bit per chan-pol -- and hence we divide by an 
        //   extra factor of 8 when considering the time axis
        maskidx_raw = (threadbaseidx / block_size + kurtblock_idx) * N_POLS;
        //   also, account for the fact that we write multiple mask blocks
        maskidx_true = (maskcounter - 1) * masksize + (maskidx_raw / 8);

        // set the element to 0 in certain cases (when we reach an integer boundary)
        //  each 8bits iterates (1 or 2) pols first, then kurtblock_index
        if (kurtblock_idx % (8 / N_POLS) == 0) {
            mask[maskidx_true] = 0;
        }

        // sum the values of the two pols and write
        mask[maskidx_true] = mask[maskidx_true] + (zap1 << (maskidx_raw % 8)) + (zap2 << ((maskidx_raw + 1) % 8));

        if (zap1 && zap2) {
            chan_start = intermediate_base;
            for (int j = chan_start; j < chan_start + block_size * N_POLS; j = j + 2) {
                replx1 = curand_normal(&state) * rstd + rmean;
                reply1 = curand_normal(&state) * istd + imean;
                // replx2 = curand_normal(&state) * rstd + rmean;
                // reply2 = curand_normal(&state) * rstd + rmean;
                asm volatile ("st.global.v4.f32 [%0], {%1, %2, %3, %4};"
                            :
                            : "l"(block + j), "f"(replx1), "f"(reply1), "f"(replx1), "f"(reply1));
            }
        }
        else if (zap1) {
            chan_start = intermediate_base;
            for (int j = chan_start; j < chan_start + block_size * N_POLS; j = j + N_POLS) {
                replx1 = curand_normal(&state) * rstd + rmean;
                reply1 = curand_normal(&state) * istd + imean;
                block[j].x = replx1;
                block[j].y = reply1;
            }
        }
        else if (zap2) {
            chan_start = intermediate_base + 1;
            for (int j = chan_start; j < chan_start + block_size * N_POLS; j = j + N_POLS) {
                replx1 = curand_normal(&state) * rstd + rmean;
                reply1 = curand_normal(&state) * istd + imean;
                block[j].x = replx1;
                block[j].y = reply1;
            }
        }
    }
}

