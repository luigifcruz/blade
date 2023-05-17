#include "blade/memory/ops.hh"

using namespace Blade;
using namespace Blade::ops::types;

template<typename IT, typename OT, U64 N>
__global__ void polarizer(const ops::complex<IT>* input,
                                ops::complex<OT>* output) {
    const int tid = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

    if (tid < (N * 2)) {
        // The complex multiplication below can be simplified because
        // the real part of the phasor is 0.0. Boring implementation:
        // const IT yPol90 = cuCmulf(yPol, make_cuFloatComplex(0.0, 1.0));

        const ops::complex<OT> yPol = input[tid + 1];
        const ops::complex<IT> xPol = input[tid];

        const ops::complex<IT> yPol90(-yPol.imag(), +yPol.real());
        
        output[tid + 0] = xPol + yPol90;
        output[tid + 1] = xPol - yPol90;
    }
}
