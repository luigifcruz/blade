#define BL_LOG_DOMAIN "PIPELINE"

#include "blade/pipeline.hh"

namespace Blade {

Pipeline::Pipeline(const U64& numberOfAccumulationSteps,
                   const U64& numberOfComputeSteps)
     : state(State::IDLE),
       numberOfAccumulationSteps(numberOfAccumulationSteps),
       numberOfComputeSteps(numberOfComputeSteps),
       accumulationStepCounter(0),
       computeStepCounter(0),
       currentComputeCount(0) {
    BL_INFO("Pipeline with {} accumulation and {} compute steps.", 
            numberOfAccumulationSteps, numberOfComputeSteps);

    BL_CUDA_CHECK_THROW(cudaStreamCreateWithFlags(&this->stream,
            cudaStreamNonBlocking), [&]{
        BL_FATAL("Failed to create stream for CUDA steam: {}", err);
    });
}

Pipeline::~Pipeline() {
    this->synchronize();
    if (this->state == State::GRAPH) {
        cudaGraphDestroy(this->graph);
    }
    cudaStreamDestroy(this->stream);
}

const U64 Pipeline::incrementAccumulatorStep() {
    return ++accumulationStepCounter;
}

const U64 Pipeline::resetAccumulatorSteps() {
    const auto& previous = accumulationStepCounter;
    accumulationStepCounter = 0;
    return previous;
}

const U64 Pipeline::incrementComputeStep() {
    return ++computeStepCounter;
}

const U64 Pipeline::resetComputeSteps() {
    const auto& previous = computeStepCounter;
    computeStepCounter = 0;
    return previous;
}

const Result Pipeline::synchronize() {
    BL_CUDA_CHECK(cudaStreamSynchronize(this->stream), [&]{
        BL_FATAL("Failed to synchronize stream: {}", err);
    });
    return Result::SUCCESS;
}

bool Pipeline::isSynchronized() {
    return cudaStreamQuery(this->stream) == cudaSuccess;
}

const Result Pipeline::compute() {
    for (auto& module : this->modules) {
        BL_CHECK(module->preprocess(this->stream, this->currentComputeCount));
    }

    switch (state) {
        case State::GRAPH:
            BL_CUDA_CHECK(cudaGraphLaunch(this->instance, this->stream), [&]{
                BL_FATAL("Failed launch CUDA graph: {}", err);
            });
            break;
        case State::CACHED:
            BL_DEBUG("Creating CUDA Graph.");
            BL_CUDA_CHECK(cudaStreamBeginCapture(this->stream,
                cudaStreamCaptureModeGlobal), [&]{
                BL_FATAL("Failed to begin the capture of CUDA Graph: {}", err);
            });

            for (auto& module : this->modules) {
                BL_CHECK(module->process(this->stream));
            }

            BL_CUDA_CHECK(cudaStreamEndCapture(this->stream, &this->graph), [&]{
                BL_FATAL("Failed to end the capture of CUDA Graph: {}", err);
            });

            BL_CUDA_CHECK(cudaGraphInstantiate(&this->instance, this->graph,
                    NULL, NULL, 0), [&]{
                BL_FATAL("Failed to instantiate CUDA Graph: {}", err);
            });

            BL_CUDA_CHECK(cudaGraphLaunch(this->instance, this->stream), [&]{
                BL_FATAL("Failed launch CUDA graph: {}", err);
            });

            this->state = State::GRAPH;
            break;
        case State::IDLE:
            // BL_DEBUG("Caching kernels ahead of CUDA Graph instantiation.");
            for (auto& module : this->modules) {
                BL_CHECK(module->process(this->stream));
            }
            // this->state = State::CACHED;
            break;
        default:
            BL_FATAL("Internal error.");
            return Result::ERROR;
    }

    BL_CUDA_CHECK_KERNEL([&]{
        BL_FATAL("Failed to process: {}", err);
        return Result::CUDA_ERROR;
    });

    this->currentComputeCount += 1;

    return Result::SUCCESS;
}

}  // namespace Blade
