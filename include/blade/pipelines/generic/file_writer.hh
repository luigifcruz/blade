#ifndef BLADE_PIPELINES_GENERIC_FILE_WRITER_HH
#define BLADE_PIPELINES_GENERIC_FILE_WRITER_HH

#include <memory>
#include <deque>

#include "blade/pipeline.hh"
#include "blade/accumulator.hh"

#include "blade/modules/guppi/writer.hh"
#include "blade/modules/filterbank/writer.hh"

namespace Blade::Pipelines::Generic {

template<typename WT, typename IT>
class BLADE_API FileWriter : public Pipeline, public Accumulator {
 public:
    struct Config {
        WT::Config writerConfig;

        ArrayTensorDimensions inputDimensions;
        BOOL transposeBTPF = false;
        U64 accumulateRate = 1;
    };

    explicit FileWriter(const Config& config);

    constexpr const U64 getStepInputBufferSize() const {
        return this->config.inputDimensions.size();
    }

    constexpr const U64 getTotalInputBufferSize() const {
        return this->writerBuffer.dims().size();
    }

    constexpr const Config& getConfig() const {
        return this->config;
    }

    const Result accumulate(const ArrayTensor<Device::CUDA, IT>& data,
                            const cudaStream_t& stream);

    const std::shared_ptr<WT> getWriter() {
        return this->writer;
    }

 private:
    const Config config;

    ArrayTensor<Device::CPU, IT> writerBuffer;

    std::shared_ptr<WT> writer;
};

}  // namespace Blade::Pipelines::Generic

#endif
