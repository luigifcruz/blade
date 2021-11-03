#include "blade/pipeline.hh"
#include <nvtx3/nvToolsExt.h>

namespace Blade {

Pipeline::~Pipeline() {
    BL_DEBUG("Deallocating device memory.");
    for (const auto& allocation : allocations) {
        cudaFree(allocation);
    }

    BL_DEBUG("Destroying CUDA assets.");
    cudaGraphDestroy(graph);
    cudaStreamDestroy(cudaStream);
}

Result Pipeline::synchronize() {
    BL_CUDA_CHECK(cudaStreamSynchronize(cudaStream), [&]{
        BL_FATAL("Failed to synchronize stream: {}", err);
    });
    return Result::SUCCESS;
}

Result Pipeline::setup() {
    BL_DEBUG("Pipeline commiting.");

    BL_DEBUG("Creating CUDA Stream.");
    BL_CUDA_CHECK(cudaStreamCreateWithFlags(&cudaStream, cudaStreamNonBlocking), [&]{
        BL_FATAL("Failed to create stream for CUDA Graph: {}", err);
    });

    BL_CHECK(this->setupModules());
    BL_CHECK(this->setupMemory());
    BL_CHECK(this->setupReport(resources));

    state = 1;
    BL_INFO("Pipeline successfully commited!");

    return Result::SUCCESS;
}

Result Pipeline::loop(const bool& async) {
    if (this->isSyncronized() == false) {
        BL_FATAL("Pipeline is not synchronized.");
        return Result::ERROR;
    }
    this->synchronized = false;
    BL_CHECK(this->loopPreprocess());
    BL_CHECK(this->loopUpload());

    switch (state) {
        case 3:
            BL_CUDA_CHECK(cudaGraphLaunch(instance, cudaStream), [&]{
                BL_FATAL("Failed launch CUDA graph: {}", err);
            });
            break;
        case 2:
            BL_DEBUG("Creating CUDA Graph.");
            BL_CUDA_CHECK(cudaStreamBeginCapture(cudaStream,
                        cudaStreamCaptureModeGlobal), [&]{
                BL_FATAL("Failed to begin the capture of CUDA Graph: {}", err);
            });

            BL_CHECK(this->loopProcess(cudaStream));

            BL_CUDA_CHECK_KERNEL([&]{
                BL_FATAL("Failed to run kernels while capturing: {}", err);
                return Result::CUDA_ERROR;
            });

            BL_CUDA_CHECK(cudaStreamEndCapture(cudaStream, &graph), [&]{
                BL_FATAL("Failed to end the capture of CUDA Graph: {}", err);
            });

            BL_CUDA_CHECK(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0), [&]{
                BL_FATAL("Failed to instantiate CUDA Graph: {}", err);
            });

            state = 3;
            break;
        case 1:
            BL_DEBUG("Caching kernels ahead of CUDA Graph instantiation.");
            BL_CHECK(this->loopProcess(cudaStream));
            state = 2;
            break;
        case 0:
            BL_FATAL("Pipeline not commited.");
            return Result::ERROR;
        default:
            BL_FATAL("Internal error.");
            return Result::ERROR;
    }

    BL_CHECK(this->loopDownload());
    BL_CUDA_CHECK(cudaLaunchHostFunc(cudaStream,
            this->callPostprocess, this), [&]{
        BL_FATAL("Failed to launch postprocess: {}", err);
        return Result::CUDA_ERROR;
    });

    BL_CUDA_CHECK_KERNEL([&]{
        BL_FATAL("Failed to process: {}", err);
        return Result::CUDA_ERROR;
    });

    if (!async) {
        BL_CHECK(this->synchronize());
    }

    return Result::SUCCESS;
}

void CUDART_CB Pipeline::callPostprocess(void* data) {
    auto pipeline = static_cast<Pipeline*>(data);

    if (pipeline->loopPostprocess() != Result::SUCCESS) {
        abort();
    }

    pipeline->synchronized = true;
}

}  // namespace Blade
