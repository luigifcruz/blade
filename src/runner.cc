#define BL_LOG_DOMAIN "RUNNER"

#include "blade/runner.hh"

namespace Blade {

Runner::Runner(const std::shared_ptr<Pipeline>& pipeline)
     : pipeline(pipeline), numberOfStreams(pipeline->numberOfStreams()), headIndex(0) {
    BL_DEBUG("Initializing runner state.")
}

Runner::~Runner() {
    if (queue.size() > 0) {
        BL_WARN("Runner destroyed with {} stream left inside queue.", queue.size());
    }
}

Result Runner::enqueue(const std::function<Result(const U64& index, U64& id, const bool& willOutput)>& callback) {
    if (queue.size() == numberOfStreams) {
        std::this_thread::yield();
        return Result::RUNNER_QUEUE_FULL;
    }

    U64 id = 0;
    const bool willOutput = (pipeline->computeStepsPerCycle() == 
                                (pipeline->computeCurrentStepCount() + 1));
    BL_CHECK(callback(headIndex, id, willOutput));
    queue.push({headIndex, id});

    headIndex = (headIndex + 1) % numberOfStreams;

    return Result::SUCCESS;
}

Result Runner::dequeue(const std::function<Result(const U64& index, const U64& id)>& callback) {
    // If queue is empty, return immediately.
    if (queue.size() == 0) {
        std::this_thread::yield();
        return Result::RUNNER_QUEUE_EMPTY;
    }

    // If queue is not full, check if head is synchronized.
    if (queue.size() < numberOfStreams) {
        const auto [frontIndex, frontId] = queue.front();
        if (pipeline->isSynchronized(frontIndex)) {
            queue.pop();
            return callback(frontIndex, frontId);
        }
        return Result::RUNNER_QUEUE_NONE_AVAILABLE;
    }

    // If queue is full, wait until head is synchronized.
    const auto [frontIndex, frontId] = queue.front();
    BL_CHECK(pipeline->synchronize(frontIndex));
    queue.pop();
    return callback(frontIndex, frontId);
}

}  // namespace Blade