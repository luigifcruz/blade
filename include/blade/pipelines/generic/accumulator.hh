#ifndef BLADE_PIPELINES_GENERIC_ACCUMULATOR_HH
#define BLADE_PIPELINES_GENERIC_ACCUMULATOR_HH

#include <memory>
#include <deque>

#include "blade/pipeline.hh"

#include "blade/modules/guppi/writer.hh"
#include "blade/modules/filterbank/writer.hh"

namespace Blade::Pipelines::Generic {

template<typename ModuleType, Device Dev, typename InputType>
class BLADE_API Accumulator : public Pipeline {
 public:
    struct Config {
        ModuleType::Config moduleConfig;

        ArrayDimensions inputDimensions; // TODO break away from implicit Array restriction
        BOOL inputIsATPFNotAFTP = false; // TODO put this in the template
        BOOL transposeATPF = false;
        BOOL reconstituteBatchedDimensions = false;
        U64 accumulateRate = 1;
    };

    explicit Accumulator(const Config& config);

    constexpr const U64 getStepInputBufferSize() const {
        return this->config.inputDimensions.size();
    }

    constexpr const U64 getTotalInputBufferSize() const {
        return this->accumulationBuffer.dims().size();
    }

    constexpr const Config& getConfig() const {
        return this->config;
    }

    const Result accumulate(const ArrayTensor<Device::CUDA, InputType>& data,
                            const cudaStream_t& stream);

    const std::shared_ptr<ModuleType> getModule() {
        return this->moduleUnderlying;
    }

 private:
    const Config config;

    ArrayTensor<Dev, InputType> accumulationBuffer;

    std::shared_ptr<ModuleType> moduleUnderlying;
};

}  // namespace Blade::Pipelines::Generic

#endif
