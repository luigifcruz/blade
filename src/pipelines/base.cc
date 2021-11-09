#include <cuda_runtime.h>

extern "C" {
    #include "blade/pipelines/base.h"
}

bool blade_pin_memory(void* buffer, size_t size) {
    return cudaHostRegister(buffer, size, cudaHostRegisterDefault) == cudaSuccess;
}

bool blade_use_device(int device_id) {
    return cudaSetDevice(device_id) == cudaSuccess;
}
