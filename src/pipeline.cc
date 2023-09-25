#define BL_LOG_DOMAIN "PIPELINE"

#include "blade/pipeline.hh"

namespace Blade {

Pipeline::Pipeline()
     : _commited(false),
       _computeStepCount(0),
       _computeStepsPerCycle(1),
       _computeLifetimeCycles(0) {
    BL_DEBUG("Creating new pipeline.");

    _streams.resize(2);
    for (U64 i = 0; i < _streams.size(); i++) {
        BL_CUDA_CHECK_THROW(cudaStreamCreateWithFlags(_streams[i], cudaStreamNonBlocking), [&]{
            BL_FATAL("Failed to create CUDA stream: {}", err);
        });
    }
}

void Pipeline::addModule(const std::shared_ptr<Module>& module) {
    if ((module->getTaint() & Taint::CHRONOUS) == Taint::CHRONOUS) {
        const auto& localRatio = module->getComputeRatio();
        if (localRatio > 1) {
            _computeStepsPerCycle *= localRatio;
            _computeStepRatios.push_back(localRatio);
        }
    }
    _modules.push_back(module);
}

Pipeline::~Pipeline() {
    for (U64 i = 0; i < _streams.size(); i++) {
        synchronize(i);
        cudaStreamDestroy(_streams[i]);
    }
    _computeStepRatios.clear();
    BL_DEBUG("Destroying pipeline after {} lifetime compute cycles.", _computeLifetimeCycles);
}

Result Pipeline::synchronize(const U64& index) {
    BL_CUDA_CHECK(cudaStreamSynchronize(_streams[index]), [&]{
        BL_FATAL("Failed to synchronize stream: {}", err);
    });
    return Result::SUCCESS;
}

bool Pipeline::isSynchronized(const U64& index) {
    return cudaStreamQuery(_streams[index]) == cudaSuccess;
}

Result Pipeline::commit() {
    BL_DEBUG("Commiting pipeline with {} compute steps per cycle.", _computeStepsPerCycle);

    // TODO: Validate pipeline topology with Taint (in-place modules).

    if (_computeStepRatios.size() == 0) {
        _computeStepRatios.push_back(1);
    }

    return Result::SUCCESS;
}

Result Pipeline::compute(const U64& index) {
    if (!_commited) {
        BL_CHECK(commit());
        _commited = true;
    }

    U64 localStepCount = 0;
    U64 localRatioIndex = 0;
    U64 localStepOffset = 1;

    for (auto& module : _modules) {
        localStepCount = _computeStepCount / localStepOffset % _computeStepRatios[localRatioIndex];

        const auto& result = module->process(localStepCount, _streams[index]);

        if (result == Result::PIPELINE_EXHAUSTED) {
            BL_INFO("Module finished pipeline execution at {} lifetime compute cycles.", _computeLifetimeCycles);
            return Result::PIPELINE_EXHAUSTED;
        }

        if (result != Result::SUCCESS) {
            return result;
        }

        if (module->getComputeRatio() > 1) {
            if ((localStepCount + 1) == _computeStepRatios[localRatioIndex]) {
                localStepOffset += localStepCount;
                localRatioIndex += 1;
            } else {
                break;
            }
        }
    }

    BL_CUDA_CHECK_KERNEL([&]{
        BL_FATAL("CUDA compute error: {}", err);
        return Result::CUDA_ERROR;
    });

    _computeStepCount += 1;

    if (_computeStepCount == _computeStepsPerCycle) {
        _computeStepCount = 0;
        _computeLifetimeCycles += 1;
    }

    return Result::SUCCESS;
}

}  // namespace Blade
