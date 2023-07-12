#define BL_LOG_DOMAIN "PIPELINE"

#include "blade/pipeline.hh"

namespace Blade {

Pipeline::Pipeline()
     : _commited(false),
       _computeTotalNumberOfSteps(1),
       _computeCurrentStepNumber(0),
       _computeNumberOfLifetimeCycles(0) {
    BL_DEBUG("Creating new pipeline.");
    BL_CUDA_CHECK_THROW(cudaStreamCreateWithFlags(&this->stream,
                                                  cudaStreamNonBlocking), [&]{
        BL_FATAL("Failed to create stream for CUDA steam: {}", err);
    });
}

void Pipeline::addModule(const std::shared_ptr<Module>& module) {
    if ((module->getTaint() & Taint::CHRONOUS) == Taint::CHRONOUS) {
        if (_computeTotalNumberOfSteps < module->getComputeRatio()) {
            _computeTotalNumberOfSteps = module->getComputeRatio();
        }
    }
    modules.push_back(module);
}

Pipeline::~Pipeline() {
    this->synchronize();
    cudaStreamDestroy(this->stream);
    BL_DEBUG("Destroying pipeline after {} lifetime compute cycles.", _computeNumberOfLifetimeCycles);
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

Result Pipeline::commit() {
    BL_DEBUG("Commiting pipeline with {} total number of compute steps.", _computeTotalNumberOfSteps);
        
    // TODO: Validate pipeline topology with Taint (in-place modules).
        
    return Result::SUCCESS;
}

Result Pipeline::compute() {
    if (!_commited) {
        BL_CHECK(commit());
        _commited = true;
    }

    for (auto& module : this->modules) {
        const auto& result = module->process(stream, _computeCurrentStepNumber);

        if (result == Result::SUCCESS) {
            continue;
        }

        if (result == Result::PIPELINE_CONTINUE) {
            break;
        }

        if (result == Result::PIPELINE_EXHAUSTED) {
            BL_INFO("Module finished pipeline execution at {} lifetime compute cycles.", _computeNumberOfLifetimeCycles);
            return Result::PIPELINE_EXHAUSTED;
        }
    }

    BL_CUDA_CHECK_KERNEL([&]{
        BL_FATAL("CUDA compute error: {}", err);
        return Result::CUDA_ERROR;
    });

    _computeCurrentStepNumber += 1;
    if (_computeCurrentStepNumber == _computeTotalNumberOfSteps) {
        _computeNumberOfLifetimeCycles += 1;
    }
    _computeCurrentStepNumber %= _computeTotalNumberOfSteps;

    return Result::SUCCESS;
}

}  // namespace Blade
