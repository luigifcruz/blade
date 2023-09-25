#ifndef BLADE_RUNNER_HH
#define BLADE_RUNNER_HH

#include <queue>
#include <vector>
#include <memory>

#include "blade/logger.hh"
#include "blade/pipeline.hh"
#include "blade/macros.hh"

namespace Blade {

class BLADE_API Runner : public Pipeline {
 public:
    Runner();
    ~Runner();

    Result enqueue(const std::function<Result()>& inputCallback,
                   const std::function<Result()>& outputCallback,
                   const U64& id = 0);
    Result dequeue(const std::function<Result(const U64& id)>& callback);

    template<typename DT, typename ST>
    Result copy(DT& dst, const ST& src) {
        return Copy(dst, src, stream(headIndex));
    }

    template<typename DT, typename ST>
    Result copy(Duet<DT>& dst, const ST& src) {
        return Copy(dst[headIndex], src, stream(headIndex));
    }

    template<typename DT, typename ST>
    Result copy(DT& dst, const Duet<ST>& src) {
        return Copy(dst, src.at(headIndex), stream(headIndex));
    }

    template<typename DT, typename ST>
    Result copy(Duet<DT>& dst, const Duet<ST>& src) {
        return Copy(dst[headIndex], src.at(headIndex), stream(headIndex));
    }

 private:
    U64 headIndex;
    std::queue<std::tuple<U64, U64>> queue;
};

}  // namespace Blade

#endif
