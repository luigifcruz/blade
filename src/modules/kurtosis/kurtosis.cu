#include "blade/memory/base.hh"
#include "cuComplex.h"

#define block_size 256
#define minv 0.00390625

#define nsamps_lo 255
#define nsamps_hi 257

#define quotient 1.00784313726
#define qinv 0.9922178988

// assuming M = 256, stddev = 5
#define sklim_lower 0.526881
#define sklim_upper 2.18694

// sklim lower mod: (1 / M) * (Qinv * sklower + 1)
// sklim upper mod: (1 / M) * (Qinv * skupper + 1)
#define sklim_lower_mod 0.005948362339
#define sklim_upper_mod 0.01238250395

using namespace Blade;

// CUDA kernel to compute sk_array
template<typename IT, typename OT, bool debugMode,
    int N_ANTS, int N_CHANS, int N_SAMPS, int N_POLS>
__global__ void compute_sk_array(cuFloatComplex* block, U8* mask, int maskcounter) {
    // let's assume STDDEV of 5 for now

    int masksize = N_ANTS * N_CHANS * N_SAMPS / (block_size * 4);
    // int masksize = N_ANTS * N_CHANS * (N_SAMPS / block_size) * N_POLS;

    float repl;
    if constexpr (debugMode) {
        repl = 100.0f;
    }
    else {
        repl = 0.0f;
    }

    /*
    int start = blockIdx.x * 1024 + threadIdx.x;
    int n = (N_ANTS * N_CHANS * N_SAMPS * N_POLS) / 65536;
    for (int j = start * n; j < start * n + n; j = j + 4) {
        asm volatile ("st.global.v4.f32 [%0], {%1, %2, %3, %4};"
                    :
                    : "l"(block + j), "f"(repl), "f"(repl), "f"(repl), "f"(repl));
        asm volatile ("st.global.v4.f32 [%0], {%1, %2, %3, %4};"
                    :
                    : "l"(block + j + 2), "f"(repl), "f"(repl), "f"(repl), "f"(repl));
    }
    return;
    */

    // Compute indices
    /*
    int chan = threadIdx.x;
    if (ant >= N_ANTS) {
        ant = ant - N_ANTS;
        ant = ant * 5 + chan / 32;
        if (ant >= N_ANTS) {
            return;
        }
        chan = 160 + (chan % 32);
    }
    */
    int ant = blockIdx.x;
    int chan = threadIdx.x + blockIdx.y * 160;
    if (chan >= N_CHANS) {
        return;
    }
    // int pol = threadIdx.y;
    // int init_samp_start = 256 * threadIdx.y;

    //printf("%d %d %d %.5f %.5f\n", ant, chan, N_SAMPS, sklim_upper, sklim_lower);

    // sk accumulators
    // float s[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    float v2_1, v2_2, sk_1, sk_2;
    float s1_1, s2_1, s1_2, s2_2;
    //float s1_1, s2_1, v2_1, sk_1, s1_2, s2_2, v2_2, sk_2;
    float x1, y1, x2, y2;
    // float nsamps_lo = block_size - 1.0f;
    // float nsamps_hi = block_size + 1.0f;
    // float quotient = ((nsamps_hi) / (nsamps_lo));

    int idx1, baseidx, intermediate_base;


    int threadbaseidx = (ant * N_CHANS + chan) * N_SAMPS;

    float skvals[64];
    int skind = 0;
    for (int samp_start = 0; samp_start < N_SAMPS; samp_start = samp_start + block_size) {
    // for (int samp_start = init_samp_start; samp_start < init_samp_start + 256; samp_start = samp_start + block_size) {
        // for (int pol = 0; pol < N_POLS; pol++) {
            //printf("%d %d %d %d : %d %d %d\n", N_ANTS, N_CHANS, N_SAMPS, N_POLS, ant, chan, pol);

            // zero-out sums

            /*
            asm volatile ("st.local.v4.f32 [%0], {%1, %2, %3, %4};"
                        :
                        : "l"(s), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));

            */
            s1_1 = 0;
            s2_1 = 0;

            s1_2 = 0;
            s2_2 = 0;

            // printf("here 1 %d %d %d\n", ant, chan, pol);
            // Compute s1 (sum of elements) and s2 (sum of squares)
            // int baseidx = (ant * N_CHANS + chan) * N_SAMPS + samp_start;
            baseidx = threadbaseidx + samp_start;
            // intermediate_base = baseidx * N_POLS + pol;
            intermediate_base = baseidx * N_POLS;
            idx1 = intermediate_base;
            // idx2 = intermediate_base + 1;

            for (int samp = 0; samp < block_size; samp++) {
                // int idx = (baseidx + samp) * N_POLS + pol;
                // cuFloatComplex value = block[idx];
                
                /*
                x1 = block[idx1].x;
                y1 = block[idx1].y;

                x2 = block[idx2].x;
                y2 = block[idx2].y;
                */


                asm volatile(
                    "ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
                    : "=f"(x1), "=f"(y1), "=f"(x2), "=f"(y2)
                    : "l"(block + idx1)
                    );

                // printf("\t\t%.5f %.5f\n", x, y);
                //v2 = value.x * value.x + value.y * value.y;
                
                // pol 1
                // v2_1 = block[idx1].x * block[idx1].x;
                // v2_1 = fmaf(block[idx1].y, block[idx1].y, v2_1);
                x1 = x1 * x1;
                y1 = y1 * y1;
                x2 = x2 * x2;
                y2 = y2 * y2;

                v2_1 = x1 + y1;
                v2_2 = x2 + y2;
                
                s1_1 += v2_1;
                s1_2 += v2_2;
                //s[0] += v2_1;
                //s[1] += v2_2;

                s2_1 = fmaf(v2_1, v2_1, s2_1);
                s2_2 = fmaf(v2_2, v2_2, s2_2);

                // s[2] = fmaf(v2_1, v2_1, s[2]);
                // s[3] = fmaf(v2_2, v2_2, s[3]);

                // pol 2
                // v2_2 = block[idx2].x * block[idx2].x;
                // v2_2 = fmaf(block[idx2].y, block[idx2].y, v2_2);

                idx1 = idx1 + N_POLS;
                // idx2 = idx1 + 1;
            }

            //printf("\ts1 s2 quotient bsize : %.5f %.5f %.5f %d\n", s1, s2, quotient, block_size);
            // Compute sk value
            
            // sk_1 = quotient * ((block_size * (s2_1 / (s1_1 * s1_1))) - 1.0f);
            // sk_2 = quotient * ((block_size * (s2_2 / (s1_2 * s1_2))) - 1.0f);

            sk_1 = s2_1 / (s1_1 * s1_1);
            sk_2 = s2_2 / (s1_2 * s1_2);
            // sk_1 = s[2] / (s[0] * s[0]);
            // sk_2 = s[3] / (s[1] * s[1]);
            skvals[skind++] = sk_1;
            skvals[skind++] = sk_2;
    }


    int chan_start;
    int zap1, zap2;
    skind = 0;
    int maskidx_raw;
    int maskidx_true;
    int maskidx1;

    for (int samp_start = 0; samp_start < N_SAMPS; samp_start = samp_start + block_size) {
    // for (int samp_start = init_samp_start; samp_start < init_samp_start + 256; samp_start = samp_start + block_size) {
            intermediate_base = (threadbaseidx + samp_start) * N_POLS;

            // based on sk we can zap the channel
            //printf("\tsk: %.5f\texp %.5f - %.5f\n", sk, sklim_lower, sklim_upper);

            sk_1 = skvals[skind++];
            sk_2 = skvals[skind++];
            zap1 = sk_1 < sklim_lower_mod || sk_1 > sklim_upper_mod;
            zap2 = sk_2 < sklim_lower_mod || sk_2 > sklim_upper_mod;

            maskidx_raw = ((ant * N_CHANS + chan) * (N_SAMPS / block_size) + (samp_start / block_size)) * N_POLS;
            maskidx_true = ((ant * N_CHANS + chan) * ((N_SAMPS) / (block_size * 4))) + (samp_start / (block_size * 4));

            maskidx_true = (maskcounter - 1) * masksize + (maskidx_true);
            // printf("%d %d\n", maskidx_true, masksize);
            if (samp_start % (block_size * 4) == 0) {
                mask[maskidx_true] = 0;
            }

            mask[maskidx_true] = mask[maskidx_true] + (zap1 << (maskidx_raw % 8)) + (zap2 << ((maskidx_raw + 1) % 8));

            if (zap1 && zap2) {
                // mask[maskidx1] = 1 << (maskidx1 % 8);
                // mask[maskidx1 + 1] = 1 << ((maskidx1 + 1) % 8);

                chan_start = intermediate_base;
                for (int j = chan_start; j < chan_start + block_size * N_POLS; j = j + 2) {
                    asm volatile ("st.global.v4.f32 [%0], {%1, %2, %3, %4};"
                                :
                                : "l"(block + j), "f"(repl), "f"(repl), "f"(repl), "f"(repl));
                }
            }
            else if (zap1) {
                // mask[maskidx1] = 1 << (maskidx1 % 8);

                chan_start = intermediate_base;
                for (int j = chan_start; j < chan_start + block_size * N_POLS; j = j + N_POLS) {
                    block[j].x = repl;
                    block[j].y = repl;
                }
            }
            else if (zap2) {
                // mask[maskidx1 + 1] = 1 << ((maskidx1 + 1) % 8);
                
                chan_start = intermediate_base + 1;
                for (int j = chan_start; j < chan_start + block_size * N_POLS; j = j + N_POLS) {
                    block[j].x = repl;
                    block[j].y = repl;
                }
            }

            // mask[maskidx1] = mask[maskidx1] + mask[maskidx1 + 1];
            // printf("%d %d\n", mask[maskidx1], mask[maskidx1 + 1]);
            
        // }
    }

}

