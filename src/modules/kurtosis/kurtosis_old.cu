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

/*
#include <thrust/memory.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
*/

using namespace Blade;

// organized by powers of two starting at 8
// eg index 0 is 2^(0 + 8) = 256
// and in ascending order of stddev
float SKLIM_VALS[] = {
    // STD 3, CHUNK 256
    0.698159, 1.49597,
    // STD 3, CHUNK 512
    0.775046, 1.32542,
    // STD 3, CHUNK 1024
    0.834186, 1.21695,

    // STD 4, CHUNK 256
    0.613738, 1.784,
    // STD 4, CHUNK 512
    0.711612, 1.48684,
    // STD 5, CHUNK 1024
    0.786484, 1.31218,

    // STD 5, CHUNK 256
    0.526881, 2.18694,
    // STD 5, CHUNK 512
    0.649093, 1.69044,
    // STD 5, CHUNK 1024
    0.740405, 1.42332
};


// CUDA kernel to compute sk_array
template<typename IT, typename OT, bool debugMode,
    int N_ANTS, int N_CHANS, int N_SAMPS, int N_POLS>
__global__ void compute_sk_array(
    cuFloatComplex* block,
    U8* mask) {
    // int N_ANTS, int N_CHANS, int N_SAMPS, int N_POLS) {//, int m) {

    // float sklim_lower, sklim_upper;
    // float sklim_lower_mod, sklim_upper_mod;
    // let's assume STDDEV of 5 for now

    // int block_size = 256;

    // sklim_lower = 0.526881;
    // sklim_upper = 2.18694;
    

    /*
    switch (block_size) {
        case 256:
            sklim_lower = 0.526881;
            sklim_upper = 2.18694;
            break;
        case 512:
            sklim_lower = 0.649093;
            sklim_upper = 1.69044;
            break;
        case 1024:
            sklim_lower = 0.740405;
            sklim_upper = 1.42332;
            break;
        case 2048:
            sklim_lower = 0.808641;
            sklim_upper = 1.27145;
            break;
        case 8192:
            sklim_lower = 0.898022;
            sklim_upper = 1.12164;
        default:
            break;
    }
    */

    // Compute indices
    int ant = threadIdx.x;
    int chan = blockIdx.x;
    int pol = threadIdx.y;
    // int init_samp_start = blockIdx.y * 2048;
    // printf("%d %d %d %.5f %.5f\n", ant, chan, N_SAMPS, sklim_upper, sklim_lower);

    float s1p1, s2p1, v2p1, skp1;
    float s1p2, s2p2, v2p2, skp2;
    
    // float nsamps_lo = block_size - 1.0f;
    // float nsamps_hi = block_size + 1.0f;
    // float quotient = ((nsamps_hi) / (nsamps_lo));

    float repl = debugMode * 100.0f;
    // float repl_y;

    /*
    if constexpr (debugMode) {
        repl = 100.0f;
    } else {
        repl = 0.0f;
    }
    */


    int threadbaseidx = (ant * N_CHANS + chan) * N_SAMPS;
    int samp;
    int baseidx;
    int idx1, idx2;
    int chan_start;
    int j;

    for (int samp_start = 0; samp_start < N_SAMPS; samp_start = samp_start + block_size) {
    // for (int samp_start = init_samp_start; samp_start < init_samp_start + 2048; samp_start = samp_start + block_size) {
        s1p1 = 0.0f;
        s2p1 = 0.0f;
        // s1p2 = 0.0f;
        // s2p2 = 0.0f;

        // int baseidx = (threadbaseidx + samp_start) * N_POLS + pol;
        // for (int idx = baseidx; idx < baseidx + block_size * N_POLS; idx = idx + N_POLS) {
        

        baseidx = threadbaseidx + samp_start;
        int intermediate_base = baseidx * N_POLS + pol;
        idx1 = intermediate_base;
        // idx2 = idx1 + 1;
        for (samp = 0; samp < block_size; samp++) {
            // idx = (baseidx + samp) * N_POLS + pol;
            // idx = intermediate_base + samp * N_POLS;

            // v2 = value.x * value.x + value.y * value.y;
            
            // v2 = x * x + y * y;
            // v2 = 0;
            // v2 = fmaf(block[idx].x, block[idx].x, 0);
            
            // pol 1
            v2p1 = block[idx1].x * block[idx1].x;
            v2p1 = fmaf(block[idx1].y, block[idx1].y, v2p1);
            
            s1p1 += v2p1;
            
            s2p1 = fmaf(v2p1, v2p1, s2p1);

            idx1 = idx1 + N_POLS;
            // idx2 = idx2 + N_POLS;
        }

        // Compute sk value
        skp1 = s2p1 / (s1p1 * s1p1);
        // skp2 = s2p2 / (s1p2 * s1p2);

        // sk = quotient * ((block_size * (s2 / (s1 * s1))) - 1.0f);
        
        // int zap1 = (skp1 > sklim_upper_mod || skp1 < sklim_lower_mod);
        // U8 zap2 = (skp2 > sklim_upper_mod || skp2 < sklim_lower_mod);

        // int maskidx = (((ant * N_CHANS + chan) * (N_SAMPS / block_size)) + (samp_start / block_size)) * N_POLS + pol;
        // int mod = maskidx % 8;
        // int mod1 = (maskidx + 1) % 8;
        
        // mask[maskidx] = zap1 << mod;
        // mask[maskidx + 1] = zap2 << mod1;
        if (skp1 > sklim_upper_mod || skp1 < sklim_lower_mod) {
            chan_start = baseidx * N_POLS + pol;
            for (j = chan_start; j < chan_start + block_size * N_POLS; j = j + N_POLS) {
                block[j].x = repl;
                block[j].y = repl;
            }
        }

    }

}

