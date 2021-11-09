#include "blade/pipeline.hh"
#include <nvtx3/nvToolsExt.h>

namespace Blade {

Pipeline::Pipeline(const bool& async, const bool& test) :
    asyncMode(async), testMode(test) {
    if (this->asyncMode && this->testMode) {
        BL_FATAL("Tests are only supported in synchronous mode.");
        throw Result::ERROR;
    }
}

Pipeline::~Pipeline() {
    this->synchronize();

    for (const auto& allocation : this->allocations) {
        cudaFree(allocation);
    }

    if (this->state == 3) {
        cudaGraphDestroy(this->graph);
    }

    cudaStreamDestroy(this->cudaStream);
}

Result Pipeline::synchronize() {
    BL_CUDA_CHECK(cudaStreamSynchronize(this->cudaStream), [&]{
        BL_FATAL("Failed to synchronize stream: {}", err);
    });
    return Result::SUCCESS;
}

bool Pipeline::isSyncronized() {
    return cudaStreamQuery(this->cudaStream) == cudaSuccess;
}

Result Pipeline::setup() {
    BL_DEBUG("Pipeline commiting.");

    BL_DEBUG("Creating CUDA Stream.");
    BL_CUDA_CHECK(cudaStreamCreateWithFlags(&this->cudaStream,
        cudaStreamNonBlocking), [&]{
        BL_FATAL("Failed to create stream for CUDA Graph: {}", err);
    });

    BL_CHECK(this->setupModules());
    BL_CHECK(this->setupMemory());
    BL_CHECK(this->setupReport(this->resources));

    if (!this->asyncMode) {
        BL_INFO("Working in synchronous mode.");
    }

    if (this->testMode) {
        BL_INFO("Setting-up test.")
        BL_CHECK(this->setupTest())
    }

    BL_CHECK(this->synchronize());

    this->state = 1;
    BL_INFO("Pipeline successfully commited!");

    return Result::SUCCESS;
}

Result Pipeline::loop() {
    BL_CHECK(this->loopPreprocess());
    BL_CHECK(this->loopUpload());

    switch (state) {
        case 3:
            BL_CUDA_CHECK(cudaGraphLaunch(this->instance, this->cudaStream), [&]{
                BL_FATAL("Failed launch CUDA graph: {}", err);
            });
            break;
        case 2:
            BL_DEBUG("Creating CUDA Graph.");
            BL_CUDA_CHECK(cudaStreamBeginCapture(this->cudaStream,
                cudaStreamCaptureModeGlobal), [&]{
                BL_FATAL("Failed to begin the capture of CUDA Graph: {}", err);
            });

            BL_CHECK(this->loopProcess(this->cudaStream));

            BL_CUDA_CHECK_KERNEL([&]{
                BL_FATAL("Failed to run kernels while capturing: {}", err);
                return Result::CUDA_ERROR;
            });

            BL_CUDA_CHECK(cudaStreamEndCapture(this->cudaStream, &this->graph), [&]{
                BL_FATAL("Failed to end the capture of CUDA Graph: {}", err);
            });

            BL_CUDA_CHECK(cudaGraphInstantiate(&this->instance, this->graph,
                    NULL, NULL, 0), [&]{
                BL_FATAL("Failed to instantiate CUDA Graph: {}", err);
            });

            this->state = 3;
            break;
        case 1:
            BL_DEBUG("Caching kernels ahead of CUDA Graph instantiation.");
            BL_CHECK(this->loopProcess(this->cudaStream));
            this->state = 2;
            break;
        case 0:
            BL_FATAL("Pipeline not commited.");
            return Result::ERROR;
        default:
            BL_FATAL("Internal error.");
            return Result::ERROR;
    }

    BL_CHECK(this->loopDownload());

    BL_CUDA_CHECK_KERNEL([&]{
        BL_FATAL("Failed to process: {}", err);
        return Result::CUDA_ERROR;
    });

    if (!this->asyncMode) {
        BL_CHECK(this->synchronize());

        if (this->testMode) {
            BL_CHECK(this->loopTest());
        }
    }

    return Result::SUCCESS;
}

}  // namespace Blade
