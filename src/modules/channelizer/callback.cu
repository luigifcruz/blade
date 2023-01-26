#include "blade/modules/channelizer/callback.hh"

namespace Blade {
namespace Internal {

__device__ cufftComplex CB_ConvertInputC(void *dataIn, size_t offset, void *callerInfo, void *sharedPtr) {
    return make_float2(0.0, 0.0);
}

__device__ cufftCallbackLoadC d_loadCallbackPtr = CB_ConvertInputC; 

Callback::Callback(cufftHandle& plan) {
    BL_CUDA_CHECK_THROW(cudaMemcpyFromSymbol(&h_loadCallbackPtr, 
                                             d_loadCallbackPtr, 
                                             sizeof(h_loadCallbackPtr)), [&]{
        BL_FATAL("The allocation of an cuFFT callback failed: {}", err);
    });

    cufftResult status = cufftXtSetCallback(plan, 
                                            (void**)&h_loadCallbackPtr, 
                                            CUFFT_CB_LD_COMPLEX,
                                            0);
    if (status != CUFFT_SUCCESS) {
        BL_FATAL("The configuration of an cuFFT callback failed: {0:#x}", status);
        BL_CHECK_THROW(Result::CUDA_ERROR);
    }

    BL_CUDA_CHECK_KERNEL_THROW([&]{
        BL_FATAL("Callbacks failed to install: {}", err);
        return Result::CUDA_ERROR;
    });

 }

}  // namespace Internal
}  // namespace Blade
