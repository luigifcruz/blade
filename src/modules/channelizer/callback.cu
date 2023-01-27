#include "blade/modules/channelizer/callback.hh"

namespace Blade {
namespace Internal {

__device__ cufftComplex CB_ConvertInputC(void *dataIn, size_t offset, void *callerInfo, void *sharedPtr) {
    auto& element = static_cast<cufftComplex*>(dataIn)[offset];

    if ((offset % 2) != 0){
        element.x = -element.x;
        element.y = -element.y;
    }

    return element;
}

__device__ cufftCallbackLoadC d_loadCallbackPtr = CB_ConvertInputC; 

Callback::Callback(cufftHandle& plan) {
    BL_CUDA_CHECK_THROW(cudaMemcpyFromSymbol(&h_loadCallbackPtr, 
                                             d_loadCallbackPtr, 
                                             sizeof(h_loadCallbackPtr)), [&]{
        BL_FATAL("The allocation of an cuFFT callback failed: {}", err);
    });

    BL_CUFFT_CHECK_THROW(cufftXtSetCallback(plan, 
                                            (void**)&h_loadCallbackPtr, 
                                            CUFFT_CB_LD_COMPLEX,
                                            0), [&]{
        BL_FATAL("The configuration of an cuFFT callback failed: {0:#x}", err);
    });

    BL_CUDA_CHECK_KERNEL_THROW([&]{
        BL_FATAL("Callbacks failed to install: {}", err);
        return Result::CUDA_ERROR;
    });
}

}  // namespace Internal
}  // namespace Blade
