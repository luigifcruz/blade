#ifndef BLADE_PIPELINES_GENERIC_FILE_WRITER_HH
#define BLADE_PIPELINES_GENERIC_FILE_WRITER_HH

#include <memory>
#include <deque>

#include "blade/pipeline.hh"

#include "blade/modules/guppi/writer.hh"

namespace Blade::Pipelines::Generic {

template<typename IT>
class BLADE_API FileWriter : public Pipeline {
 public:
    struct Config {
        std::string outputGuppiFile;
        bool directio;

        ArrayShape inputShape;
        U64 accumulateRate;

        U64 writerBlockSize = 512;
    };

    explicit FileWriter(const Config& config);

    constexpr void headerPut(std::string key, std::string value) {
        return guppi->headerPut(key, value);
    }

    constexpr void headerPut(std::string key, F64 value) {
        return guppi->headerPut(key, value);
    }

    constexpr void headerPut(std::string key, I64 value) {
        return guppi->headerPut(key, value);
    }

    constexpr void headerPut(std::string key, I32 value) {
        return guppi->headerPut(key, value);
    }

    constexpr void headerPut(std::string key, U64 value) {
        return guppi->headerPut(key, value);
    }

    constexpr const U64 getStepInputBufferSize() const {
        return this->config.inputShape.size();
    }

    constexpr const U64 getTotalInputBufferSize() const {
        return this->writerBuffer.size();
    }

    constexpr const Config& getConfig() const {
        return this->config;
    }

    const Result accumulate(const ArrayTensor<Device::CUDA, IT>& data,
                            const cudaStream_t& stream);

 private:
    const Config config;

    ArrayTensor<Device::CPU, IT> writerBuffer;

    using GuppiWriter = typename Modules::Guppi::Writer<IT>;
    std::shared_ptr<GuppiWriter> guppi;
};

}  // namespace Blade::Pipelines::Generic

#endif
