#include "blade/memory/base.hh"
#include "cuComplex.h"

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
template<typename IT, typename OT, bool debugMode>
__global__ void compute_sk_array(
    cuFloatComplex* block,
    int N_ANTS, int N_CHANS, int N_SAMPS, int N_POLS) {//, int m) {

    float sklim_lower, sklim_upper;
    // let's assume STDDEV of 5 for now

    int block_size = 256;

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

    // Compute indices
    int ant = threadIdx.x;    // Antenna index
    int chan = blockIdx.x; // blockIdx.y;   // Channel index
    int pol = threadIdx.y;   // Polarization index

    //printf("%d %d %d %.5f %.5f\n", ant, chan, N_SAMPS, sklim_upper, sklim_lower);

    float s1, s2, v2, sk, x, y;
    float nsamps_lo = block_size - 1.0f;
    float nsamps_hi = block_size + 1.0f;
    float quotient = ((nsamps_hi) / (nsamps_lo));

    // const int n = 256 * 192;

    // thrust::host_vector<int> re_arr(256 * 192);
    // thrust::host_vector<int> im_arr(256 * 192);

    // float* re_arr = (float*)malloc(sizeof(float) * n);
    // float* im_arr = (float*)malloc(sizeof(float) * n);
    // int med_arr_ind = 0;

    for (int samp_start = 0; samp_start < N_SAMPS; samp_start = samp_start + block_size) {
    // for (int pol = 0; pol < N_POLS; pol++) {
        if (ant < N_ANTS && chan < N_CHANS) {
            //printf("%d %d %d %d : %d %d %d\n", N_ANTS, N_CHANS, N_SAMPS, N_POLS, ant, chan, pol);

            // zero-out sums
            s1 = 0.0f;
            s2 = 0.0f;


            // printf("here 1 %d %d %d\n", ant, chan, pol);
            // Compute s1 (sum of elements) and s2 (sum of squares)
            int baseidx = (ant * N_CHANS + chan) * N_SAMPS + samp_start;
            for (int samp = 0; samp < block_size; samp++) {
                int idx = (baseidx + samp) * N_POLS + pol;
                cuFloatComplex value = block[idx];
                x = value.x;
                y = value.y;
                // printf("\t\t%.5f %.5f\n", x, y);
                //v2 = value.x * value.x + value.y * value.y;
                v2 = x * x + y * y;
                s1 += v2;
                s2 += v2 * v2;
            }

            //printf("\ts1 s2 quotient bsize : %.5f %.5f %.5f %d\n", s1, s2, quotient, block_size);
            // Compute sk value
            sk = quotient * ((block_size * (s2 / (s1 * s1))) - 1.0f);
            
            // based on sk we can zap the channel
            //printf("\tsk: %.5f\texp %.5f - %.5f\n", sk, sklim_lower, sklim_upper); 
            //if (1 == 1) {
            if (sk > sklim_upper || sk < sklim_lower) {
                // compute block median
                /*
                med_arr_ind = 0;

                int medianbase = ant * N_CHANS * N_SAMPS * N_POLS;
                int timeoffset = samp_start * N_POLS + pol;
                int chanprod = N_SAMPS * N_POLS;

                // float x, y;

                for (int chanidx = 0; chanidx < N_CHANS; chanidx++) {
                    int median_baseind = medianbase + (chanidx * chanprod) + timeoffset;
                    for (int samp = 0; samp < block_size; samp++) {
                        //int ind = medianbase + (chanidx * chanprod) + timeoffset + samp * N_POLS;
                        median_baseind += N_POLS;
                        // x = block[median_baseind].x;
                        // y = block[median_baseind].y;

                        re_arr[med_arr_ind] = block[median_baseind].x;
                        im_arr[med_arr_ind] = block[median_baseind].y;
                        med_arr_ind++;
                    }
                }
                */

                // thrust::sort(re_arr, re_arr + n);
                // thrust::sort(im_arr, im_arr + n);
                // heapSort(re_arr, n);
                // heapSort(im_arr, n);

                // TODO
                float repl_x;
                float repl_y;
                // float repl_x = re_arr[n / 2];
                // float repl_y = im_arr[n / 2];

                if constexpr (debugMode) {
                    repl_x = 100.0f;
                    repl_y = 100.0f;
                } else {
                    repl_x = 0.0f;
                    repl_y = 0.0f;
                }
                
                int chan_start = baseidx * N_POLS + pol;
                for (int j = chan_start; j < chan_start + block_size * N_POLS; j = j + N_POLS) {
                    block[j].x = repl_x;
                    block[j].y = repl_y;
                }
            }
        }
    }

    // free(re_arr);
    // free(im_arr);
}

/*
// Host function to call the kernel
template<typename IT, typename OT>
__global__ void get_sk_array(
    cuFloatComplex* d_block,
    int N_ANTS, int N_CHANS, int N_SAMPS, int N_POLS) {//, int m) {

    dim3 gridDim(N_ANTS, N_CHANS);    // One block per antenna and channel
    dim3 blockDim(N_POLS);           // One thread per polarization

    compute_sk_array<<<gridDim, blockDim>>>(
        d_block, N_ANTS, N_CHANS, N_SAMPS, N_POLS);//, m);
}
*/
