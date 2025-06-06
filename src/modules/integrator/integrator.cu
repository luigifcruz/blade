#include "blade/memory/base.hh"

using namespace Blade;

template<typename IT, typename OT, U64 integrationSize, U64 numberOfIntegratedElements, U64 numberOfElements>
__global__ void integrator(const ArrayTensor<Device::CUDA, IT> input,
                                 ArrayTensor<Device::CUDA, OT> output) {
    const U64 tid = (blockIdx.x * blockDim.x + threadIdx.x);

    if (tid < numberOfElements) {
        OT accumulator[numberOfIntegratedElements] = {};

        for (U64 i = 0; i < integrationSize; i++) {
            for (U64 j = 0; j < numberOfIntegratedElements; j++) {
                accumulator[j] += input[(tid * integrationSize * numberOfIntegratedElements) + (i * numberOfIntegratedElements) + j];
            }
        }

        for (U64 j = 0; j < numberOfIntegratedElements; j++) {
            output[(tid * numberOfIntegratedElements) + j] += accumulator[j];
        }
    }
}
