#include "blade/memory/base.hh"

using namespace Blade;

// Performance wise, this is a very bad CUDA kernel.
// Several improvements can be made to it.
template<typename T, typename S, U64 Axis, U64 Offset>
__global__ void accumulate(const Vector<Device::CUDA, T, S> input,
                                 Vector<Device::CUDA, T, S> output) {
    const U64 tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < input.size()) {
        output[output.shape().offsetToOffset<Axis, Offset>(tid)] = input[tid];
    }
}