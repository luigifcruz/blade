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
    BL_CUDA_CHECK(cudaStreamCreateWithFlags(&cudaStream,
                cudaStreamNonBlocking), [&]{
        BL_FATAL("Failed to create stream for CUDA Graph: {}", err);
    });

    BL_DEBUG("Setup underlying module.");
    BL_CHECK(this->underlyingInit());
    BL_CHECK(this->underlyingAllocate());
    BL_CHECK(this->underlyingReport(resources));
    BL_CHECK(this->underlyingPreprocess());

    BL_DEBUG("Pre-caching CUDA kernels.");
    BL_CHECK(this->underlyingProcess(cudaStream));

    BL_DEBUG("Creating CUDA Graphs.");
    BL_CUDA_CHECK(cudaStreamBeginCapture(cudaStream,
                cudaStreamCaptureModeGlobal), [&]{
        BL_FATAL("Failed to begin the capture of CUDA Graph: {}", err);
    });

    BL_CHECK(this->underlyingProcess(cudaStream));

    BL_CUDA_CHECK_KERNEL([&]{
        BL_FATAL("Failed to run kernels while capturing: {}", err);
        return Result::CUDA_ERROR;
    });

    BL_CUDA_CHECK(cudaStreamEndCapture(cudaStream,& graph), [&]{
        BL_FATAL("Failed to end the capture of CUDA Graph: {}", err);
    });

    BL_CUDA_CHECK(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0), [&]{
        BL_FATAL("Failed to instantiate CUDA Graph: {}", err);
    });

    BL_INFO("Pipeline commited successfully.");

    return Result::SUCCESS;
}

Result Pipeline::process(bool waitCompletion) {
    BL_CHECK(this->underlyingPreprocess());

    BL_CUDA_CHECK(cudaGraphLaunch(instance, cudaStream), [&]{
        BL_FATAL("Failed launch CUDA graph: {}", err);
    });

    if (waitCompletion) {
        cudaStreamSynchronize(cudaStream);
    }

    BL_CHECK(this->underlyingPostprocess());

    BL_CUDA_CHECK_KERNEL([&]{
        BL_FATAL("Failed to run CUDA Graph: {}", err);
        return Result::CUDA_ERROR;
    });

    return Result::SUCCESS;
}

} // namespace Blade
