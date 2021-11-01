#ifndef BLADE_CUDA_H
#define BLADE_CUDA_H

#include "blade/common.hh"

namespace Blade {

enum class RegisterKind : unsigned int {
    Mapped = cudaHostRegisterMapped,
    ReadOnly = cudaHostRegisterReadOnly,
    Default = cudaHostRegisterDefault,
    Portable = cudaHostRegisterPortable,
};

enum class CopyKind : unsigned int {
    D2H = cudaMemcpyDeviceToHost,
    H2D = cudaMemcpyHostToDevice,
    D2D = cudaMemcpyDeviceToDevice,
    H2H = cudaMemcpyHostToHost,
};

}  // namespace Blade

#endif  // BLADE_INCLUDE_BLADE_CUDA_HH_
