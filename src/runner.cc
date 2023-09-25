#define BL_LOG_DOMAIN "RUNNER"

#include "blade/runner.hh"

namespace Blade {

Runner::Runner() : headIndex(0) {
    BL_DEBUG("Initializing runner state.")
}

Runner::~Runner() {
    if (queue.size() > 0) {
        BL_WARN("Runner destroyed with {} stream left inside queue.", queue.size());
    }
}

Result Runner::enqueue(const std::function<Result()>& inputCallback,
                       const std::function<Result()>& outputCallback,
                       const U64& id) {
    if (queue.size() == numberOfStreams()) {
        std::this_thread::yield();
        return Result::RUNNER_QUEUE_FULL;
    }

    const bool willOutput = (computeStepsPerCycle() == (computeCurrentStepCount() + 1));
    BL_TRACE("[E] Index: {} | Id: {} | Will Output: {}", headIndex, id, willOutput ? "Y" : "N");

    BL_CHECK(inputCallback());
    BL_CHECK(compute(headIndex));
    if (willOutput) {
        BL_CHECK(outputCallback());
    }
    queue.push({headIndex, id});
    headIndex = (headIndex + 1) % numberOfStreams();

    return Result::SUCCESS;
}

Result Runner::dequeue(const std::function<Result(const U64& id)>& callback) {
    // If queue is empty, return immediately.
    if (queue.size() == 0) {
        std::this_thread::yield();
        return Result::RUNNER_QUEUE_EMPTY;
    }

    // If queue is not full, check if head is synchronized.
    if (queue.size() < numberOfStreams()) {
        const auto [frontIndex, frontId] = queue.front();
        if (isSynchronized(frontIndex)) {
            queue.pop();
            BL_TRACE("[D] Index: {} | Id: {}", frontIndex, frontId);
            return callback(frontId);
        }
        return Result::RUNNER_QUEUE_NONE_AVAILABLE;
    }

    // If queue is full, wait until head is synchronized.
    const auto [frontIndex, frontId] = queue.front();
    BL_CHECK(synchronize(frontIndex));
    queue.pop();
    BL_TRACE("[D] Index: {} | Id: {}", frontIndex, frontId);
    return callback(frontId);
}

}  // namespace Blade
