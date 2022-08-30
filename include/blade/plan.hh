#ifndef BLADE_PLAN_HH
#define BLADE_PLAN_HH

#include <deque>
#include <vector>
#include <memory>

#include "blade/logger.hh"
#include "blade/module.hh"
#include "blade/pipeline.hh"
#include "blade/macros.hh"
#include "blade/accumulator.hh"

namespace Blade {

class BLADE_API Plan {

 public:
    static void Available(auto& runner) {
        // Check if runner has an available slot.
        if (!runner->slotAvailable()) {
            BL_CHECK_THROW(Result::PLAN_SKIP_NO_SLOT);
        }
    } 

    static void Dequeue(auto& runner, U64* id) {
        // Dequeue job from runner.
        if (!runner->dequeue(id)) {
            BL_CHECK_THROW(Result::PLAN_SKIP_NO_DEQUEUE);
        }
    }

    static bool Loop() {
        // Prevent memory clobber inside spin-loop.
        __builtin_ia32_pause();

        // Continue with the loop.
        return true;
    }

    // Compute is used to trigger the compute step of a pipeline.
    template<class T>
    static void Compute(T& pipeline) {
        // If pipeline has accumulator, check if it's complete.
        if constexpr (std::is_base_of<Accumulator, T>::value) {
            if (!pipeline.accumulationComplete()) {
                BL_CHECK_THROW(Result::PLAN_SKIP_ACCUMULATION_INCOMPLETE);
            }
        }

        // Run compute step.
        BL_CHECK_THROW(pipeline.compute());

        // If pipeline has accumulator, reset it after compute.
        if constexpr (std::is_base_of<Accumulator, T>::value) {
            pipeline.resetAccumulatorSteps();
        }
    } 

    // TransferIn is used to the copy data to a pipeline.
    template<typename... Args>
    static void TransferIn(auto& pipeline, Args&... transfers) {
        // Check if destionation pipeline is synchronized.
        if (!pipeline.isSynchronized()) {
            BL_CHECK_THROW(Result::PLAN_ERROR_DESTINATION_NOT_SYNCHRONIZED);     
        }

        // Transfer data to the pipeline.
        BL_CHECK_THROW(pipeline.transferIn(transfers..., pipeline.getCudaStream()));
    }

    // TransferOut(3) is used to transfer output data from one pipeline to a vector.
    template<Device SD, typename ST, Device DD, typename DT>
    static void TransferOut(Vector<SD, ST>& dst, const Vector<DD, DT>& src, auto& pipeline) {
        // Transfer data to the vector.
        BL_CHECK_THROW(Memory::Copy(dst, src, pipeline.getCudaStream()));
    }

    // TransferOut(3, ...) is used to transfer output data from one pipeline to another.
    template<typename... Args>
    static void TransferOut(auto& destinationRunner, auto& sourceRunner, Args&... transfers) {
        // Check if runner has an available slot.
        if (!destinationRunner->slotAvailable()) {
            BL_CHECK_THROW(Result::PLAN_ERROR_NO_SLOT);
        }

        // Fetch runners pipelines.
        auto& sourcePipeline = sourceRunner->getWorker(sourceRunner->getHead());
        auto& destinationPipeline = destinationRunner->getNextWorker();

        // Check if destionation pipeline is synchronized.
        if (!destinationPipeline.isSynchronized()) {
            BL_CHECK_THROW(Result::PLAN_ERROR_DESTINATION_NOT_SYNCHRONIZED);     
        }

        // Fetch CUDA stream of source pipeline.
        const auto& stream = sourcePipeline.getCudaStream();

        // Transfer data to the pipeline.
        BL_CHECK_THROW(destinationPipeline.transferIn(transfers..., stream));
    } 

    // Accumulate is used to concatenate output data from one pipeline to another.
    template<typename T, typename... Args>
    static void Accumulate(T& destinationRunner, auto& sourceRunner, Args&... transfers) {
        // Check if runner supports accumulation.
        if constexpr (std::is_base_of<Accumulator, T>::value) {
            BL_CHECK_THROW(Result::PLAN_ERROR_NO_ACCUMULATOR);
        }

        // Check if runner has an available slot.
        if (!destinationRunner->slotAvailable()) {
            BL_CHECK_THROW(Result::PLAN_ERROR_NO_SLOT);
        }

        // Fetch runners pipelines.
        auto& sourcePipeline = sourceRunner->getWorker(sourceRunner->getHead());
        auto& destinationPipeline = destinationRunner->getNextWorker();

        // Check if destionation pipeline is synchronized.
        if (!destinationPipeline.isSynchronized()) {
            BL_CHECK_THROW(Result::PLAN_ERROR_DESTINATION_NOT_SYNCHRONIZED);     
        }

        // Check if pipeline is not full.
        if (destinationPipeline.accumulationComplete()) {
            BL_CHECK_THROW(Result::PLAN_ERROR_ACCUMULATION_COMPLETE);
        }

        // Fetch CUDA stream of source pipeline.
        const auto& stream = sourcePipeline.getCudaStream();

        // Transfer data to the pipeline.
        BL_CHECK_THROW(destinationPipeline.accumulate(transfers..., stream));

        // Increment pipeline accumulator.
        destinationPipeline.incrementAccumulatorStep();
    } 

    // Skip lets the user skip a cycle programmatically.
    static void Skip() {
        BL_CHECK_THROW(Result::PLAN_SKIP_USER_INITIATED);
    }

 private:
    Plan();
};

}  // namespace Blade

#endif
