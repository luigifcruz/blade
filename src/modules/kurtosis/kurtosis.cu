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
            // PFA=1.3498980316300957e-03
            sklim_lower = 0.698159;
            sklim_upper = 1.49597;
            break;
        case 4:
            // PFA=3.1671241833119958e-05
            sklim_lower = 0.613738;
            sklim_upper = 1.784;
            break;
        case 6:
            // PFA=9.8658764503770140e-10
            sklim_lower = 0.431631;
            sklim_upper = 2.769471;
            break;
        case 7:
            // PFA=1.2798125438858348e-12
            sklim_lower = 0.319825;
            sklim_upper = 3.638890;
            break;
        case 8:
            // PFA=6.2209605742718204e-16
            sklim_lower = 0.178846;
            sklim_upper = 4.703961;
            break;
        case 9:
            // PFA=1.1285884059538425e-19
            sklim_lower = -0.012557;
            sklim_upper = 4.808045;
            break;
        case 5:
        default: // default is 5
            // PFA=2.8665157187919449e-07
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
    
    // calculate stddev of real/image for both pols
    // holds square sum, and for 8bit integers and block_size=256
    // the max of (127**2)*256 is < 256**3 which is the
    // single precision integer limit... no need for doubles
    float stddev_r1, stddev_r2, stddev_i1, stddev_i2; 

    // initial fallback stddev is 0.0 until non-flagged sk-channel found
    float stddev_r1_fallback = 0.0f;
    float stddev_r2_fallback = 0.0f;
    float stddev_i1_fallback = 0.0f;
    float stddev_i2_fallback = 0.0f;

    int idx1, baseidx, intermediate_base;

    int threadbaseidx = (ant * N_CHANS + chan) * N_SAMPS;

    // float skvals[64];
    
    // for this ant-chan: one sk_val for each chan and pol
    float skvals[N_POLS * N_SAMPS / block_size];
    float stddev_rstd[N_POLS * N_SAMPS / block_size];
    float stddev_istd[N_POLS * N_SAMPS / block_size];
    int skind = 0;
    for (int samp_start = 0; samp_start < N_SAMPS; samp_start = samp_start + block_size) {
        // set accumulators to 0
        s1_1 = 0;
        s2_1 = 0;

        s1_2 = 0;
        s2_2 = 0;

        stddev_r1 = 0;
        stddev_r2 = 0;
        stddev_i1 = 0;
        stddev_i2 = 0;

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

            // sum of squared components for std-dev
            // assume mean of 0
            // except in debug leave the sum as zero for stddev of 0
            if constexpr (!debugMode) {
                stddev_r1 += x1;
                stddev_r2 += x2;
                stddev_i1 += y1;
                stddev_i2 += y2;
            }

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
        // and note the (population) std-deviation
        // except use the most recent fallback if the sk-channel will be flagged
        // as we cannot use the stddev of an out of range sample-population
        if (sk_1 < sklim_lower || sk_1 > sklim_upper) {
            stddev_rstd[skind] = stddev_r1_fallback;
            stddev_istd[skind] = stddev_i1_fallback;
        }
        else {
            stddev_rstd[skind] = sqrtf(stddev_r1 / (block_size));
            stddev_r1_fallback = stddev_rstd[skind];
            stddev_istd[skind] = sqrtf(stddev_i1 / (block_size));
            stddev_i1_fallback = stddev_istd[skind];
        }
        skvals[skind++] = sk_1;
        
        if (sk_2 < sklim_lower || sk_2 > sklim_upper) {
            stddev_rstd[skind] = stddev_r2_fallback;
            stddev_istd[skind] = stddev_i2_fallback;
        }
        else {
            stddev_rstd[skind] = sqrtf(stddev_r2 / (block_size));
            stddev_r2_fallback = stddev_rstd[skind];
            stddev_istd[skind] = sqrtf(stddev_i2 / (block_size));
            stddev_i2_fallback = stddev_istd[skind];
        }
        skvals[skind++] = sk_2;
    }


    int chan_start;
    int zap1, zap2;
    skind = 0;
    int maskidx_raw;
    int maskidx_true;

    float2 repl1, repl2;
    float rmean;
    float imean;
    float rstd_1, rstd_2;
    float istd_1, istd_2;

    // modify distributions based on if we are debugging
    if constexpr (debugMode) {
        rmean = 100.0f;
        imean= 100.0f;
    } else {
        rmean = 0.0f;
        imean = 0.0f;
    }


    curandState state;
    // ant * chan + ant just creates a unique id for each thread
    curand_init(1234ULL, ant * chan + ant, 0, &state);

    for (int kurtblock_idx = 0; kurtblock_idx < N_SAMPS/block_size; kurtblock_idx++) {
        intermediate_base = (threadbaseidx + kurtblock_idx*block_size) * N_POLS;

        // based on sk we can zap the channel
        // use the observed stddev, unless it is zero as in the case of initial fallbacks,
        // then use latest fallback (hopefully not also zero)
        rstd_1 = stddev_rstd[skind] == 0.0 ? stddev_r1_fallback : stddev_rstd[skind];
        istd_1 = stddev_istd[skind] == 0.0 ? stddev_i1_fallback : stddev_istd[skind];
        sk_1 = skvals[skind++];

        rstd_2 = stddev_rstd[skind] == 0.0 ? stddev_r2_fallback : stddev_rstd[skind];
        istd_2 = stddev_istd[skind] == 0.0 ? stddev_i2_fallback : stddev_istd[skind];
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
                repl1 = curand_normal2(&state);
                repl1.x *= rstd_1 + rmean;
                repl1.y *= istd_1 + imean;
                repl2 = curand_normal2(&state);
                repl2.x *= rstd_2 + rmean;
                repl2.y *= istd_2 + imean;
                asm volatile ("st.global.v4.f32 [%0], {%1, %2, %3, %4};"
                            :
                            : "l"(block + j), "f"(repl1.x), "f"(repl1.y), "f"(repl2.x), "f"(repl2.y));
            }
        }
        else if (zap1) {
            chan_start = intermediate_base;
            for (int j = chan_start; j < chan_start + block_size * N_POLS; j = j + N_POLS) {
                repl1 = curand_normal2(&state);
                repl1.x *= rstd_1 + rmean;
                repl1.y *= istd_1 + imean;
                block[j].x = repl1.x;
                block[j].y = repl1.y;
            }
        }
        else if (zap2) {
            chan_start = intermediate_base + 1;
            for (int j = chan_start; j < chan_start + block_size * N_POLS; j = j + N_POLS) {
                repl2 = curand_normal2(&state);
                repl2.x *= rstd_2 + rmean;
                repl2.y *= istd_2 + imean;
                block[j].x = repl2.x;
                block[j].y = repl2.y;
            }
        }
    }
}
