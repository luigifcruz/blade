#ifndef BLADE_PIPELINES_GENERIC_ACCUMULATE_HH
#define BLADE_PIPELINES_GENERIC_ACCUMULATE_HH

#include <memory>
#include <deque>

#include "blade/pipeline.hh"
#include "blade/accumulator.hh"

#include "blade/modules/guppi/writer.hh"
#include "blade/modules/filterbank/writer.hh"

namespace Blade::Pipelines::Generic {

template<typename MT, Device DT, typename IT>
class BLADE_API Accumulate : public Pipeline, public Accumulator {
 public:
    struct Config {
        MT::Config moduleConfig;

        ArrayTensorDimensions inputDimensions;
        BOOL transposeATPF = false;
        BOOL reconstituteBatchedDimensions = false;
        U64 accumulateRate = 1;
    };

    explicit Accumulate(const Config& config);

    constexpr const U64 getStepInputBufferSize() const {
        return this->config.inputDimensions.size();
    }

    constexpr const U64 getTotalInputBufferSize() const {
        return this->accumulationBuffer.dims().size();
    }

    constexpr const Config& getConfig() const {
        return this->config;
    }

    const Result accumulate(const ArrayTensor<Device::CUDA, IT>& data,
                            const cudaStream_t& stream);

    const std::shared_ptr<MT> getModule() {
        return this->moduleUnderlying;
    }

 private:
    const Config config;

    ArrayTensor<DT, IT> accumulationBuffer;

    std::shared_ptr<MT> moduleUnderlying;
};

}  // namespace Blade::Pipelines::Generic

#endif
