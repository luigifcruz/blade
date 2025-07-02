#include "blade/memory/base.hh"
#include "cuComplex.h"
#include <curand_kernel.h>

// #define block_size 256
// #define minv 0.00390625

// #define nsamps_lo 255
// #define nsamps_hi 257

// #define quotient 1.00784313726
// #define qinv 0.9922178988

// assuming M = 256, stddev = 5
// #define sklim_lower 0.526881
// #define sklim_upper 2.18694

// sklim lower mod: (1 / M) * (Qinv * sklower + 1)
// sklim upper mod: (1 / M) * (Qinv * skupper + 1)
// #define sklim_lower_mod 0.005948362339
// #define sklim_upper_mod 0.01238250395

using namespace Blade;

// CUDA kernel to compute sk_array
template<typename IT, typename OT, bool debugMode,
    int N_ANTS, int N_CHANS, int N_SAMPS, int N_POLS,
    int nKurtosisSigma, int block_size>
__global__ void compute_sk_array(cuFloatComplex* block, U8* mask, int maskcounter) {
    // let's assume STDDEV of 5 for now

    constexpr float normDistVals[512] = {7.301,  -2.929,   0.772,   2.299,   0.939,   2.671,  -0.659,
        -7.25 ,   0.359,  -6.216,  -3.133,  -3.789,   4.505,   8.52 ,   2.869,   8.52 ,   1.622,
         1.702,  10.392,  14.706,   0.438,  -8.44 ,   9.484,  -2.651,  -1.876, -11.116,  -1.968,
         3.795,  -7.428,   9.108,   7.864,   2.466,  -7.13 ,  -8.547, -12.429,  -2.547, -10.686,
         0.484,   2.611,  -2.295, -16.328,  -8.335,  -1.892,  -5.573,  -9.41 , -13.914, -14.487,
        -8.071,   1.87 ,  13.031, -13.696, -14.921,   6.307, -11.536,  -7.321,  18.455,  -0.101,
        14.814,   1.302,   8.017,  -2.051,  -8.803,  -3.233,  -3.653,   0.76 ,  -0.643, -10.814,
         9.36 , -12.117,  -2.826,   4.012,  -4.56 ,  -2.934, -15.222,  12.446,   1.753,  -7.7  ,
        -4.208,   2.025,   2.279,   7.564,   7.327,   1.849,  -0.976,  20.498,   4.629,   1.923,
         1.028,  -3.456,  -3.38 ,   5.13 ,   8.797, -10.658,   3.764,  14.62 ,  -2.126,   8.239,
       -10.355,   2.528,   9.269, -13.903,  -9.027,  -3.193,   9.014,   2.154,   1.623, -11.394,
        10.29 ,   9.14 ,  -3.229,   3.802,   7.381,  -4.059,  -7.178,   6.904,   5.695, -10.779,
        27.535,   1.086,   6.102,  -7.515,  -0.939,  -8.835,  -6.674,  -5.308,   4.889,  -2.191,
         3.491,   5.917,   3.434,  10.209,  -7.965,  -1.156,   3.781,   4.284,  -8.798,  -3.572,
         8.929,   3.689,  13.534,   0.191,  17.99 ,  13.487,  -3.778,  -0.757,   0.029,   5.907,
        -1.901,   4.673, -11.081, -11.704,  20.902,  10.315,   7.414,   0.82 ,  -5.877,   2.023,
         2.652,   8.338,   4.294,   4.791, -10.486,   5.962,   6.884,  -2.678,  -2.255,  -2.985,
        -5.511,  -8.804,   6.251,  -1.009,  -1.46 ,  -0.68 ,   8.701,  -4.504,  -6.563,   0.108,
         7.363,  10.325,  -7.82 ,  -4.421,  -7.077,   3.754,  -5.763, -12.066,  -3.121,   7.248,
        -7.884,  -0.173,   6.794,  -8.191,  -2.664, -10.414,  -7.385,  12.699,  -2.661,  -1.996,
         9.151,  -4.131,   0.537,  -9.05 ,   4.055,  -8.89 ,  -3.167,   3.988,  -7.474,  -0.346,
        -9.212,  10.706,   6.746,   6.183,   5.476,   1.185,  -7.761,   7.51 , -10.705,   6.513,
        -4.19 ,   4.429,  -8.311, -11.51 ,   9.858,  -7.14 ,   9.477, -10.458,   5.279,  -4.371,
         3.591,  -7.984,  -4.679,  -9.387,  -5.412, -14.898,  -0.96 ,  -9.642, -10.208,  -2.032,
         3.627,   7.817,  -5.036, -22.673,  -0.955,  -0.261,   4.768,   1.013,  -4.41 ,  -3.176,
         2.218,  -4.305, -10.127,   2.462,  -2.982,   5.665, -14.301,   3.622,   5.396, -12.325,
       -16.623,   0.219,  -5.294,   2.6  ,   3.57 ,  -3.142,   5.448, -22.374,  -7.723,  10.947,
        -7.959,   2.903,   9.456,  -5.262,   8.787,  14.924,  -1.06 ,   4.841, -13.498,  -4.662,
        -4.133,   1.726,   9.997,  -1.628,  14.807,  -4.257,   4.893,   5.288,   0.413,  -9.946,
         7.116,  -7.206,  -3.47 ,  -5.372,   1.808,  -5.608,   4.041,   7.849,   4.022,  -1.241,
         5.73 ,  -2.286,   3.847,  -8.876, -14.   ,  -2.6  ,  -3.298,   0.429,  -0.075,  15.849,
        -2.873,   4.936,   6.755,   3.282,  10.8  ,   1.167,  -8.888,  -9.816,   4.705,  15.64 ,
         7.347,   7.298, -10.204,  -8.299,   3.51 ,   1.491,  -6.749,  -8.146,  -7.07 ,   5.325,
        -6.325,  -9.363,   6.346,  10.622,   9.402,   6.452,   7.121,  -1.755, -14.24 ,  15.471,
        -2.855,  -8.245,   5.398,  10.135,   2.274,   0.011, -13.741, -11.671,  -3.965,  -4.094,
        -0.852,   4.926,  -2.186,   5.799,  -1.513,   8.741,  -1.812,   0.231,  -1.757,  -9.743,
       -16.323,  -7.64 , -14.721, -11.528,  -5.699,  -0.701,   2.649,  -5.948,  -6.014,   1.63 ,
         4.271,  -2.605,  10.033,  -3.449,  -3.924,   1.165,  -0.077,  -2.946,  -2.818,   6.114,
       -11.935,  -3.126,  -0.685,   7.817,  -2.295, -13.132,  -7.244,   1.183,  -2.316,  15.055,
         2.943,  -8.289,  -6.619,  10.718,   2.058,   1.824,  18.44 ,   2.741,  -4.221,  -9.308,
         3.041,   5.333,  -0.768,  -3.199,   2.647,   6.591,  11.246,  12.111,   7.237,   3.192,
         1.29 ,   9.809,   2.212,   3.628,   2.919,   4.128,   3.762,  -1.011,  -3.699, -17.048,
         1.966,  10.184,  -3.445,   7.839,   7.141,  15.263,  20.681,  11.926,  11.821,   7.432,
        -7.306,  -5.822,   7.51 ,   7.467,   9.896,   2.681,  11.445,  -7.44 ,  -5.395,   4.17 ,
        -3.413,  -2.659, -16.089,  -1.375,  10.138,  -0.96 ,   5.439,   0.25 ,   6.741,  -9.538,
         3.596,   2.85 ,  22.336,  -7.584, -20.052,  10.192,  10.021,   9.777,  -0.902,  10.84 ,
       -11.225,  10.598,  -0.996,  -9.815,   9.254,   1.755, -10.424,   7.093,   2.936,  -1.131,
         9.169,   9.878,  -3.637,  -4.086,  -0.674,   7.009,  -4.74 ,   5.697,   2.297,  -1.427,
         8.942,  -1.866,  -8.952,   6.029,  -3.8  ,  -4.557,  -3.794,   9.721,  -2.925,  -2.411,
         5.665,   5.073, -17.285,  -0.286,  -9.638,   9.828,  -2.696,   0.424,  -4.304, -12.316,
       -15.943,   6.543,   1.482,  -9.038,  -0.637, -11.345,   2.4  , -14.811,   2.669,  13.424,
         2.087, -10.742,  -9.386,  -3.076,  -3.658
    };

    constexpr int masksize = N_ANTS * N_CHANS * N_SAMPS / (block_size * 4);
    // constexpr int nsamps_lo = block_size - 1;
    // constexpr int nsamps_hi = block_size + 1;

    constexpr double quotient = (1.0 * block_size + 1) / (1.0 * block_size - 1);
    // constexpr double quotient = 1.00784313726;

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

    float replx, reply;

    if constexpr (debugMode) {
        replx = 100.0f;
        reply = 100.0f;
    } else {
        replx = 0.0f;
        reply = 0.0f;
    }
    /*
    else {
        replx = 0.0f;
        reply = 0.0f;
    }
    */

    // Compute indices
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
            
            sk_1 = quotient * ((block_size * (s2_1 / (s1_1 * s1_1))) - 1.0f);
            sk_2 = quotient * ((block_size * (s2_2 / (s1_2 * s1_2))) - 1.0f);

            // sk_1 = s2_1 / (s1_1 * s1_1);
            // sk_2 = s2_2 / (s1_2 * s1_2);
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

    double rmean = 0.0;
    double imean = 0.0;
    double rstd = 8.0f;
    double istd = 8.0f;

    /*
    for (int a = ant * N_CHANS * N_SAMPS * N_POLS; a < (ant + 1) * N_CHANS * N_SAMPS * N_POLS; a++) {
        rvar = rvar + pow(block[a].x - rmean, static_cast<double>(2.0));
        ivar = ivar + pow(block[a].y - imean, static_cast<double>(2.0));
    }

    rvar = rvar / (N_CHANS * N_SAMPS * N_POLS);
    ivar = ivar / (N_CHANS * N_SAMPS * N_POLS);

    rstd = pow(rvar, static_cast<double>(0.5));
    istd = pow(ivar, static_cast<double>(0.5));
    */

    curandState state;
    // ant * chan + ant just creates a unique id for each thread
    curand_init(1234ULL, ant * chan + ant, 0, &state);

    for (int samp_start = 0; samp_start < N_SAMPS; samp_start = samp_start + block_size) {
    // for (int samp_start = init_samp_start; samp_start < init_samp_start + 256; samp_start = samp_start + block_size) {
            intermediate_base = (threadbaseidx + samp_start) * N_POLS;

            // based on sk we can zap the channel
            //printf("\tsk: %.5f\texp %.5f - %.5f\n", sk, sklim_lower, sklim_upper);

            sk_1 = skvals[skind++];
            sk_2 = skvals[skind++];
            // zap1 = sk_1 < sklim_lower_mod || sk_1 > sklim_upper_mod;
            // zap2 = sk_2 < sklim_lower_mod || sk_2 > sklim_upper_mod;
            zap1 = sk_1 < sklim_lower || sk_1 > sklim_upper;
            zap2 = sk_2 < sklim_lower || sk_2 > sklim_upper;

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
                    if constexpr (!debugMode) {
                        replx = normDistVals[(chan + j) % 512];
                        reply = normDistVals[(ant + j) % 512];
                        // replx = curand_normal(&state) * rstd + rmean;
                        // reply = curand_normal(&state) * istd + imean;
                    }
                    asm volatile ("st.global.v4.f32 [%0], {%1, %2, %3, %4};"
                                :
                                : "l"(block + j), "f"(replx), "f"(reply), "f"(replx), "f"(reply));
                }
            }
            else if (zap1) {
                // mask[maskidx1] = 1 << (maskidx1 % 8);

                chan_start = intermediate_base;
                for (int j = chan_start; j < chan_start + block_size * N_POLS; j = j + N_POLS) {
                    if constexpr (!debugMode) {
                        replx = normDistVals[(chan + j) % 512];
                        reply = normDistVals[(ant + j) % 512];
                        // replx = curand_normal(&state) * rstd + rmean;
                        // reply = curand_normal(&state) * istd + imean;
                    }
                    block[j].x = replx;
                    block[j].y = reply;
                }
            }
            else if (zap2) {
                // mask[maskidx1 + 1] = 1 << ((maskidx1 + 1) % 8);
                
                chan_start = intermediate_base + 1;
                for (int j = chan_start; j < chan_start + block_size * N_POLS; j = j + N_POLS) {
                    if constexpr (!debugMode) {
                        replx = normDistVals[(chan + j) % 512];
                        reply = normDistVals[(ant + j) % 512];
                        // replx = curand_normal(&state) * rstd + rmean;
                        // reply = curand_normal(&state) * istd + imean;
                    }
                    block[j].x = replx;
                    block[j].y = reply;
                }
            }

            // mask[maskidx1] = mask[maskidx1] + mask[maskidx1 + 1];
            // printf("%d %d\n", mask[maskidx1], mask[maskidx1 + 1]);
            
        // }
    }

}

