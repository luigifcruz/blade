#include "cuComplex.h"
#include <stdint.h>

#include "blade/memory/ops.hh"

using namespace Blade::ops::types;

__device__ cuFloatComplex detect(const cuFloatComplex a) {
    return make_cuFloatComplex((a.x * a.x) + (a.y * a.y), 0.0f);
}

template<U64 NBEAMS, U64 NANTS, U64 NCHANS,
         U64 NTIME,  U64 NPOLS, U64 TBLOCK,
         BOOL EnableIncoherentBeam, BOOL EnableIncoherentBeamSqrt>
__global__ void ATA(const cuFloatComplex* input,
                    const cuFloatComplex* phasor,
                          cuFloatComplex* out) {
    int bi = threadIdx.x;
    int ti = bi + (blockIdx.y * TBLOCK);
    int ch = blockIdx.x;

    // Load the phasors to shared memory.
    __shared__ cuFloatComplex phr_cache[NBEAMS][NANTS][NPOLS];

    int iy = (ch * NPOLS) + (bi * NPOLS * NCHANS * NANTS);
    const int dy = NPOLS * NCHANS;

    if (bi < NBEAMS) {
        for (int a = 0; a < NANTS; a++, iy += dy) {
            phr_cache[bi][a][0] = phasor[iy+0];
            phr_cache[bi][a][1] = phasor[iy+1];
        }
    }

    __syncthreads();

    // Load the antenna values to registers.
    cuFloatComplex ant_cache[NANTS][NPOLS];

    int ix = (ch * NTIME * NPOLS) + (ti * NPOLS);
    const int dx = NTIME * NCHANS * NPOLS;

    for (int a = 0; a < NANTS; a++, ix += dx) {
        ant_cache[a][0] = input[ix+0];
        ant_cache[a][1] = input[ix+1];
    }

    // Multiply and accumulate.
    int iz = (ch * NTIME) + ti;
    const int dz = NTIME * NCHANS;

    for (int b = 0; b < NBEAMS; b++, iz += dz) {
        cuFloatComplex acc[NPOLS] = {{0.0, 0.0}};

        for (int a = 0; a < NANTS; a++) {
            acc[0] = cuCaddf(acc[0], cuCmulf(ant_cache[a][0], phr_cache[b][a][0]));
            acc[1] = cuCaddf(acc[1], cuCmulf(ant_cache[a][1], phr_cache[b][a][1]));
        }

        reinterpret_cast<float4*>(out)[iz] = *reinterpret_cast<float4*>(acc);
    }

    if (EnableIncoherentBeam) {
        cuFloatComplex acc[NPOLS] = {{0.0, 0.0}};

        for (int a = 0; a < NANTS; a++) {
            acc[0] = cuCaddf(acc[0], detect(cuCmulf(ant_cache[a][0], phr_cache[0][a][0])));
            acc[1] = cuCaddf(acc[1], detect(cuCmulf(ant_cache[a][1], phr_cache[0][a][1])));
        }

        if (EnableIncoherentBeamSqrt) {
            acc[0] = make_cuFloatComplex(sqrt(acc[0].x), acc[0].y);
            acc[1] = make_cuFloatComplex(sqrt(acc[1].x), acc[1].y);
        }

        reinterpret_cast<float4*>(out)[iz] = *reinterpret_cast<float4*>(acc);
    }
}

template<U64 NBEAMS, U64 NANTS, U64 NCHANS,
         U64 NTIME, U64 NPOLS, U64 TBLOCK,
         BOOL EnableIncoherentBeam, BOOL EnableIncoherentBeamSqrt>
__global__ void MeerKAT(const cuFloatComplex* input,
                        const cuFloatComplex* phasor,
                              cuFloatComplex* out) {
    int bi = threadIdx.x;
    int ti = bi + (blockIdx.y * TBLOCK);
    int ch = blockIdx.x;

    // Load the antenna values to registers.
    cuFloatComplex ant_cache[NANTS][NPOLS];

    int ix = (ch * NTIME * NPOLS) + (ti * NPOLS);
    const int dx = NTIME * NCHANS * NPOLS;

    for (int a = 0; a < NANTS; a++, ix += dx) {
        ant_cache[a][0] = input[ix+0];
        ant_cache[a][1] = input[ix+1];
    }

    // Multiply and accumulate.
    int iy = 0;
    int iz = (ch * NTIME) + ti;
    const int dz = NTIME * NCHANS;

    for (int b = 0; b < NBEAMS; b++, iz += dz) {
        cuFloatComplex acc[NPOLS] = {{0.0, 0.0}};

        for (int a = 0, x = ix; a < NANTS; a++, iy += 1, x += dx) {
            acc[0] = cuCaddf(acc[0], cuCmulf(ant_cache[a][0], phasor[iy]));
            acc[1] = cuCaddf(acc[1], cuCmulf(ant_cache[a][1], phasor[iy]));
        }

        reinterpret_cast<float4*>(out)[iz] = *reinterpret_cast<float4*>(acc);
    }
}
