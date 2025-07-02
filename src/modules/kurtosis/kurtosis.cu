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
    int chan = threadIdx.x + blockIdx.y * 160;
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
            s1_1 = 0;
            s2_1 = 0;

            s1_2 = 0;
            s2_2 = 0;

            baseidx = threadbaseidx + samp_start;
            intermediate_base = baseidx * N_POLS;
            idx1 = intermediate_base;

            for (int samp = 0; samp < block_size; samp++) {
                
                asm volatile(
                    "ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
                    : "=f"(x1), "=f"(y1), "=f"(x2), "=f"(y2)
                    : "l"(block + idx1)
                    );

                x1 = x1 * x1;
                y1 = y1 * y1;
                x2 = x2 * x2;
                y2 = y2 * y2;

                v2_1 = x1 + y1;
                v2_2 = x2 + y2;
                
                s1_1 += v2_1;
                s1_2 += v2_2;

                s2_1 = fmaf(v2_1, v2_1, s2_1);
                s2_2 = fmaf(v2_2, v2_2, s2_2);

                idx1 = idx1 + N_POLS;
            }

            // Compute sk value
            sk_1 = quotient * ((block_size * (s2_1 / (s1_1 * s1_1))) - 1.0f);
            sk_2 = quotient * ((block_size * (s2_2 / (s1_2 * s1_2))) - 1.0f);

            skvals[skind++] = sk_1;
            skvals[skind++] = sk_2;
    }


    int chan_start;
    int zap1, zap2;
    skind = 0;
    int maskidx_raw;
    int maskidx_true;
    // int maskidx1;

    float replx, reply;
    double rmean;
    double imean;
    double rstd;
    double istd;

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

    for (int samp_start = 0; samp_start < N_SAMPS; samp_start = samp_start + block_size) {
        intermediate_base = (threadbaseidx + samp_start) * N_POLS;

        // based on sk we can zap the channel

        sk_1 = skvals[skind++];
        sk_2 = skvals[skind++];
        
        zap1 = sk_1 < sklim_lower || sk_1 > sklim_upper;
        zap2 = sk_2 < sklim_lower || sk_2 > sklim_upper;

        // maskidx_raw corresponds to the element in a mask with no int-to-bit
        // reduction
        // maskidx_true corresponds to the element in a mask where 8 ints are
        // reduced into one -- one bit per chan-pol -- and hence we divide by an 
        // extra factor of 8 when considering the time axis
        maskidx_raw = ((ant * N_CHANS + chan) * (N_SAMPS / block_size) + (samp_start / block_size)) * N_POLS;
        maskidx_true = (((ant * N_CHANS + chan) * ((N_SAMPS) / (block_size * 8))) + (samp_start / (block_size * 8))) * N_POLS;

        // account for the fact that we write multiple mask blocks
        maskidx_true = (maskcounter - 1) * masksize + (maskidx_true);

        // set the element to 0 in certain cases (when we reach an integer boundary)
        if (samp_start % (block_size * (8 / N_POLS)) == 0) {
            mask[maskidx_true] = 0;
        }

        // sum the values of the two pols and write
        mask[maskidx_true] = mask[maskidx_true] + (zap1 << (maskidx_raw % 8)) + (zap2 << ((maskidx_raw + 1) % 8));

        if (zap1 && zap2) {
            chan_start = intermediate_base;
            for (int j = chan_start; j < chan_start + block_size * N_POLS; j = j + 2) {
                // replx = normDistVals[(chan + j) % 512];
                // reply = normDistVals[(ant + j) % 512];
                replx = curand_normal(&state) * rstd + rmean;
                reply = curand_normal(&state) * istd + imean;
                asm volatile ("st.global.v4.f32 [%0], {%1, %2, %3, %4};"
                            :
                            : "l"(block + j), "f"(replx), "f"(reply), "f"(replx), "f"(reply));
            }
        }
        else if (zap1) {
            chan_start = intermediate_base;
            for (int j = chan_start; j < chan_start + block_size * N_POLS; j = j + N_POLS) {
                // replx = normDistVals[(chan + j) % 512];
                // reply = normDistVals[(ant + j) % 512];
                replx = curand_normal(&state) * rstd + rmean;
                reply = curand_normal(&state) * istd + imean;
                block[j].x = replx;
                block[j].y = reply;
            }
        }
        else if (zap2) {
            chan_start = intermediate_base + 1;
            for (int j = chan_start; j < chan_start + block_size * N_POLS; j = j + N_POLS) {
                // replx = normDistVals[(chan + j) % 512];
                // reply = normDistVals[(ant + j) % 512];
                replx = curand_normal(&state) * rstd + rmean;
                reply = curand_normal(&state) * istd + imean;
                block[j].x = replx;
                block[j].y = reply;
            }
        }
    }
}

