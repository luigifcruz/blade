#include "blade/modules/channelizer/callback.hh"

namespace Blade {
namespace Internal {

__device__ cufftComplex CB_ConvertInputC_2POL(void *dataIn, size_t offset, void *callerInfo, void *sharedPtr) {
    auto element = static_cast<cufftComplex*>(dataIn)[offset];

    if ((offset % 4) != 0){
        element.x = -element.x;
        element.y = -element.y;
    }

    return element;
}

__managed__ __device__ cufftCallbackLoadC d_2POL_loadCallbackPtr = CB_ConvertInputC_2POL; 

__device__ cufftComplex CB_ConvertInputC_1POL(void *dataIn, size_t offset, void *callerInfo, void *sharedPtr) {
    auto element = static_cast<cufftComplex*>(dataIn)[offset];

    if ((offset % 2) != 0){
        element.x = -element.x;
        element.y = -element.y;
    }

    return element;
}

__managed__ __device__ cufftCallbackLoadC d_1POL_loadCallbackPtr = CB_ConvertInputC_1POL; 

__device__ cufftComplex CB_ConvertInputC(void *dataIn, size_t offset, void *callerInfo, void *sharedPtr) {
    const auto& inverseStride = static_cast<uint64_t*>(callerInfo)[0];
    auto element = static_cast<cufftComplex*>(dataIn)[offset];

    if ((offset % inverseStride) != 0){
        element.x = -element.x;
        element.y = -element.y;
    }

    return element;
}

__managed__ __device__ cufftCallbackLoadC d_loadCallbackPtr = CB_ConvertInputC; 

Callback::Callback(cufftHandle& plan, const uint64_t& numberOfPolarizations) {
    void** callerInfoPtr = nullptr;
    void** callbackPointer = nullptr;

    if (numberOfPolarizations == 1) {
        callbackPointer = (void**)&d_1POL_loadCallbackPtr;
    } else if (numberOfPolarizations == 2) {
        callbackPointer = (void**)&d_2POL_loadCallbackPtr;
    } else {
        BL_CUDA_CHECK_THROW(cudaMallocManaged(&callerInfo, sizeof(uint64_t)), [&]{
            BL_FATAL("Can't allocate CUDA memory: {}", err);
        });
        this->callerInfo[0] = static_cast<uint64_t>(numberOfPolarizations) * 2;

        callerInfoPtr = (void**)&callerInfo;
        callbackPointer = (void**)&d_loadCallbackPtr;
    }

    BL_CUFFT_CHECK_THROW(cufftXtSetCallback(plan, 
                                            callbackPointer, 
                                            CUFFT_CB_LD_COMPLEX,
                                            callerInfoPtr), [&]{
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
