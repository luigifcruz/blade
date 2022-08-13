#ifndef BLADE_ACCUMULATOR_HH
#define BLADE_ACCUMULATOR_HH

#include <span>
#include <string>
#include <memory>
#include <vector>

#include "blade/logger.hh"
#include "blade/module.hh"

namespace Blade {

class BLADE_API Accumulator {
 public:
    const bool accumulationComplete() const;

 protected:
    explicit Accumulator(const U64& numberOfSteps); 

    const U64 getAccumulatorNumberOfSteps() const;
    const U64 getCurrentAccumulatorStep() const;
    const U64 incrementAccumulatorStep();
    const U64 resetAccumulatorSteps();

 private:
    const U64 numberOfSteps;
    U64 stepCounter;
};

}  // namespace Blade

#endif
