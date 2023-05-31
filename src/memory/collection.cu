#include "blade/memory/base.hh"

using namespace Blade;

__global__ void accumulate() {
    const int tid = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

 //   if (tid < input.size()) {
 //   }
}
