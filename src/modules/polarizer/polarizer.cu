#include <cuComplex.h>
#include <cuda_fp16.h>
#include <stdint.h>

template<typename IT, typename OT, uint64_t N>
__global__ void polarizer(const IT* input, OT* output) {
    const int tid = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

    if (tid < (N * 2)) {
        // The complex multiplication below can be simplified because
        // the real part of the phasor is 0.0. Boring implementation:
        // const IT yPol90 = cuCmulf(yPol, make_cuFloatComplex(0.0, 1.0));

        const OT& yPol = input[tid + 1];
        const float x = -cuCimagf(yPol);
        const float y = +cuCrealf(yPol);
        const IT yPol90 = make_cuFloatComplex(x, y);

        const IT xPol = input[tid + 0];
        output[tid + 0] = cuCaddf(xPol, yPol90);
        output[tid + 1] = cuCsubf(xPol, yPol90);
    }
}
