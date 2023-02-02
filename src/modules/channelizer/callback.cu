#include "blade/modules/channelizer/callback.hh"

namespace Blade {
namespace Internal {

__device__ cufftComplex CB_ConvertInputC(void *dataIn, size_t offset, void *callerInfo, void *sharedPtr) {
    const auto& numberOfPolarizations = static_cast<U64*>(callerInfo)[0];
    auto element = static_cast<cufftComplex*>(dataIn)[offset];

    if (((offset / numberOfPolarizations) % 2) != 0){
        element.x = -element.x;
        element.y = -element.y;
    }

    return element;
}

__managed__ __device__ cufftCallbackLoadC d_loadCallbackPtr = CB_ConvertInputC; 

Callback::Callback(cufftHandle& plan, const U64& numberOfPolarizations) {
    BL_CUDA_CHECK_THROW(cudaMallocManaged(&callerInfo, sizeof(U64)), [&]{
        BL_FATAL("Can't allocate CUDA memory: {}", err);
    });
    this->callerInfo[0] = static_cast<U64>(numberOfPolarizations);

    BL_CUFFT_CHECK_THROW(cufftXtSetCallback(plan, 
                                            (void**)&d_loadCallbackPtr, 
                                            CUFFT_CB_LD_COMPLEX,
                                            (void**)&callerInfo), [&]{
        BL_FATAL("The configuration of an cuFFT callback failed: {0:#x}", err);
    });

    BL_CUDA_CHECK_KERNEL_THROW([&]{
        BL_FATAL("Callbacks failed to install: {}", err);
        return Result::CUDA_ERROR;
    });
}

Callback::~Callback() {
    if (callerInfo) {
        cudaFree(callerInfo);
    }
}

}  // namespace Internal
}  // namespace Blade
