#include "blade/pipeline.hh"

namespace Blade {

Pipeline::~Pipeline() {
    if (!this->commited) {
        return;
    }

    cudaGraphDestroy(graph);
    cudaStreamDestroy(cudaStream);
}

Result Pipeline::commit() {
    BL_INFO("Pipeline commiting...");

    // Let child class allocate memory.
    BL_INFO("Allocating memory...");
    BL_CHECK(this->underlyingAllocate());

    BL_CUDA_CHECK(cudaStreamCreateWithFlags(&cudaStream,
                cudaStreamNonBlocking), [&]{
        BL_FATAL("Failed to create stream for CUDA Graph: {}", err);
    });

    // Run kernels once to populate cache.
    BL_INFO("Pre-caching CUDA kernels...");
    BL_CHECK(this->underlyingProcess(cudaStream));

    BL_INFO("Creating CUDA Graphs...");
    BL_CUDA_CHECK(cudaStreamBeginCapture(cudaStream,
                cudaStreamCaptureModeGlobal), [&]{
        BL_FATAL("Failed to begin the capture of CUDA Graph: {}", err);
    });

    BL_CHECK(this->underlyingProcess(cudaStream));

    BL_CUDA_CHECK(cudaStreamEndCapture(cudaStream, &graph), [&]{
        BL_FATAL("Failed to end the capture of CUDA Graph: {}", err);
    });

    BL_CUDA_CHECK(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0), [&]{
        BL_FATAL("Failed to instantiate CUDA Graph: {}", err);
    });

    commited = true;
    BL_INFO("Pipeline commited successfully!");

    return Result::SUCCESS;
}

Result Pipeline::process(bool waitCompletion) {
    BL_CUDA_CHECK(cudaGraphLaunch(instance, 0), [&]{
        BL_FATAL("Failed launch CUDA graph: {}", err);
    });

    if (waitCompletion) {
        cudaStreamSynchronize(cudaStream);
    }

    return Result::SUCCESS;
}

} // namespace Blade
