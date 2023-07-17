#include "blade/memory/base.hh"

using namespace Blade;

// TODO: Several improvements can be made to this kernel.
template<typename T>
__global__ void permutation(const ArrayTensor<Device::CUDA, T> input,
                                  ArrayTensor<Device::CUDA, T> output,
                                  ArrayShape permutationIndex) {
    const U64 tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < input.size()) {
        ArrayShape::Type originalCoords = input.shape().offsetToShape(tid);
        for (U64 dim = 0; dim < permutationIndex.dimensions(); dim++) {
            permutedCoords[dim] = originalCoords[permutationIndex[dim]];
        }
        output[permutedCoords] = input[tid];
    }
}