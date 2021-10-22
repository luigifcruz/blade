#include "cuComplex.h"

template<size_t NBEAMS, size_t NANTS, size_t NCHANS,
         size_t NTIME, size_t NPOLS, size_t TBLOCK>
__global__ void ATA(const char2* in,
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

    int ix = (ch * NTIME) + (ti);
    const int dx = NTIME * NCHANS;

    for (int a = 0; a < NANTS; a++, ix += dx) {
        const char4 tmp = reinterpret_cast<const char4*>(in)[ix];
        ant_cache[a][0] = make_cuFloatComplex(tmp.x, tmp.y);
        ant_cache[a][1] = make_cuFloatComplex(tmp.z, tmp.w);
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
}

template<size_t NBEAMS, size_t NANTS, size_t NCHANS,
         size_t NTIME, size_t NPOLS, size_t TBLOCK>
__global__ void MeerKAT(const char2* in,
                        const cuFloatComplex* phasor,
                              cuFloatComplex* out) {
    int bi = threadIdx.x;
    int ti = bi + (blockIdx.y * TBLOCK);
    int ch = blockIdx.x;

    // Load the antenna values to registers.
    cuFloatComplex ant_cache[NANTS][NPOLS];

    int ix = (ch * NTIME) + (ti);
    const int dx = NTIME * NCHANS;

    for (int a = 0; a < NANTS; a++, ix += dx) {
        const char4 tmp = reinterpret_cast<const char4*>(in)[ix];
        ant_cache[a][0] = make_cuFloatComplex(tmp.x, tmp.y);
        ant_cache[a][1] = make_cuFloatComplex(tmp.z, tmp.w);
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
