#include "blade/accumulator.hh"

namespace Blade {

Accumulator::Accumulator(const U64& numberOfSteps) 
     : numberOfSteps(numberOfSteps),
       stepCounter(0) {
    BL_INFO("New accumulator of {} steps.", numberOfSteps);
}

const U64 Accumulator::getAccumulatorNumberOfSteps() const {
    return numberOfSteps;
}

const U64 Accumulator::getCurrentAccumulatorStep() const {
    return stepCounter;
}

const bool Accumulator::accumulationComplete() const {
    return stepCounter == numberOfSteps;
}

const U64 Accumulator::incrementAccumulatorStep() {
    return ++stepCounter;
}

const U64 Accumulator::resetAccumulatorSteps() {
    const auto& previous = stepCounter;
    stepCounter = 0;
    return previous;
}

}  // namespace Blade
