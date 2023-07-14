#include "blade/memory/base.hh"

using namespace Blade;

// TODO: Several improvements can be made to this kernel.
template<typename T, U64 Axis, U64 Offset>
__global__ void accumulate(const ArrayTensor<Device::CUDA, T> input,
                                 ArrayTensor<Device::CUDA, T> output) {
    const U64 tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < input.size()) {
        output[output.shape().offsetToOffset<Axis, Offset>(tid)] = input[tid];
    }
}