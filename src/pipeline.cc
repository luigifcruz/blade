#define BL_LOG_DOMAIN "PIPELINE"

#include "blade/pipeline.hh"

namespace Blade {

Pipeline::Pipeline(const U64& numberOfComputeSteps)
     : numberOfComputeSteps(numberOfComputeSteps),
       currentComputeStep(0),
       lifetimeComputeCycles(0) {
    BL_INFO("New pipeline with {} compute steps.", numberOfComputeSteps);

    BL_CUDA_CHECK_THROW(cudaStreamCreateWithFlags(&this->stream,
                                                  cudaStreamNonBlocking), [&]{
        BL_FATAL("Failed to create stream for CUDA steam: {}", err);
    });
}

Pipeline::~Pipeline() {
    this->synchronize();
    cudaStreamDestroy(this->stream);
}

Result Pipeline::synchronize() {
    BL_CUDA_CHECK(cudaStreamSynchronize(this->stream), [&]{
        BL_FATAL("Failed to synchronize stream: {}", err);
    });
    return Result::SUCCESS;
}

bool Pipeline::isSynchronized() {
    return cudaStreamQuery(this->stream) == cudaSuccess;
}

// TODO: Add skip logic.
Result Pipeline::compute() {
    for (auto& module : this->modules) {
        BL_CHECK(module->process(stream, currentComputeStep));
    }

    BL_CUDA_CHECK_KERNEL([&]{
        BL_FATAL("Failed to process: {}", err);
        return Result::CUDA_ERROR;
    });

    currentComputeStep = (currentComputeStep + 1) % numberOfComputeSteps;
    lifetimeComputeCycles += 1;

    return Result::SUCCESS;
}

}  // namespace Blade
