#include "blade/pipeline.hh"

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

Result Pipeline::commit() {
    BL_DEBUG("Pipeline commiting.");

    BL_DEBUG("Creating CUDA Stream.");
    BL_CUDA_CHECK(cudaStreamCreateWithFlags(&cudaStream, cudaStreamNonBlocking), [&]{
        BL_FATAL("Failed to create stream for CUDA Graph: {}", err);
    });

    BL_CHECK(this->underlyingInit());
    BL_CHECK(this->underlyingAllocate());
    BL_CHECK(this->underlyingReport(resources));

    state = 1;
    BL_INFO("Pipeline successfully commited!");

    return Result::SUCCESS;
}

void CUDART_CB Pipeline::handlePostprocess(void* data) {
    auto pipeline = static_cast<Pipeline*>(data);

    if (pipeline->underlyingPostprocess() != Result::SUCCESS) {
        BL_FATAL("Postprocess function emitted an error.");
        abort();
    }
}

Result Pipeline::handleProcess() {
    BL_CHECK(this->underlyingProcess(cudaStream));
    BL_CUDA_CHECK(cudaLaunchHostFunc(cudaStream, this->handlePostprocess, this), [&]{
        BL_FATAL("Failed to launch postprocess while capturing: {}", err);
        return Result::CUDA_ERROR;
    });

    return Result::SUCCESS;
}

Result Pipeline::process(bool waitCompletion) {
    BL_CHECK(this->underlyingPreprocess());

    switch (state) {
        case 3:
            BL_CUDA_CHECK(cudaGraphLaunch(instance, cudaStream), [&]{
                BL_FATAL("Failed launch CUDA graph: {}", err);
            });
            break;
        case 2:
            BL_DEBUG("Creating CUDA Graphs.");
            BL_CUDA_CHECK(cudaStreamBeginCapture(cudaStream,
                        cudaStreamCaptureModeGlobal), [&]{
                BL_FATAL("Failed to begin the capture of CUDA Graph: {}", err);
            });

            BL_CHECK(this->handleProcess());

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
            BL_CHECK(this->handleProcess());
            state = 2;
            break;
        case 0:
            BL_FATAL("Pipeline not commited.");
            return Result::ERROR;
        default:
            BL_FATAL("Internal error.");
            return Result::ERROR;
    }

    if (waitCompletion) {
        cudaStreamSynchronize(cudaStream);
    }

    BL_CUDA_CHECK_KERNEL([&]{
        BL_FATAL("Failed to process: {}", err);
        return Result::CUDA_ERROR;
    });

    return Result::SUCCESS;
}

}  // namespace Blade
