#ifndef BLADE_RUNNER_HH
#define BLADE_RUNNER_HH

#include <queue>
#include <vector>
#include <memory>

#include "blade/logger.hh"
#include "blade/pipeline.hh"
#include "blade/macros.hh"

namespace Blade {

class BLADE_API Runner {
 public:
    Runner(const std::shared_ptr<Pipeline>& pipeline);
    ~Runner();

    Result enqueue(const std::function<Result(const U64& index, U64& id, const bool& willOutput)>& callback);
    Result dequeue(const std::function<Result(const U64& index, const U64& id)>& callback);

 private:
    std::queue<std::tuple<U64, U64>> queue;
    std::shared_ptr<Pipeline> pipeline;
    U64 numberOfStreams;
    U64 headIndex;
};

}  // namespace Blade

#endif
