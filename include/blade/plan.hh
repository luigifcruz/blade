#ifndef BLADE_PLAN_HH
#define BLADE_PLAN_HH

#include <deque>
#include <thread>
#include <vector>
#include <memory>

#include "blade/logger.hh"
#include "blade/runner.hh"
#include "blade/module.hh"
#include "blade/pipeline.hh"
#include "blade/macros.hh"

namespace Blade {

class BLADE_API Plan {

 public:
    template<class T>
    static void Available(const std::unique_ptr<Runner<T>>& runner) {
        // Check if runner has an available slot.
        if (!runner->slotAvailable()) {
            BL_CHECK_THROW(Result::PLAN_SKIP_NO_SLOT);
        }

        // Get hot pipeline.
        auto& pipeline = runner->getNextWorker();

        // Check if pipeline is ready to output.
        if (pipeline.computeComplete()) {
            BL_CHECK_THROW(Result::PLAN_SKIP_NO_SLOT);
        }

        // Check if pipeline is not processing.
        if (!pipeline.isSynchronized()) {
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
        std::this_thread::yield();

        // Continue with the loop.
        return true;
    }

    // Compute is used to trigger the compute step of a pipeline.
    template<class Pipeline>
    static void Compute(Pipeline& pipeline) {
        // Run compute step.
        BL_CHECK_THROW(pipeline.compute());
    } 

    // TransferIn(3) is used to transfer data to a pipeline input vector.
    template<Device SDev, Device DDev, typename Type, typename Shape>
    static void TransferIn(Vector<SDev, Type, Shape>& dst, 
                           const Vector<DDev, Type, Shape>& src, 
                           auto& pipeline) {
        // Check if destionation pipeline is synchronized.
        if (!pipeline.isSynchronized()) {
            pipeline.synchronize();
        }

        // Transfer data to the vector.
        BL_CHECK_THROW(Memory::Copy(dst, src, pipeline.getCudaStream()));
    }

    // TransferOut(3) is used to transfer data from a pipeline output vector.
    template<Device SDev, Device DDev, typename Type, typename Shape>
    static void TransferOut(Vector<SDev, Type, Shape>& dst, 
                            const Vector<DDev, Type, Shape>& src, 
                            auto& pipeline) {
        // Check if pipeline is ready to output.
        if (pipeline.computeComplete()) {
            BL_CHECK_THROW(Result::PLAN_SKIP_NO_SLOT);
        }

        // Transfer data to the vector.
        BL_CHECK_THROW(Memory::Copy(dst, src, pipeline.getCudaStream()));
    }

 private:
    Plan();
};

}  // namespace Blade

#endif
