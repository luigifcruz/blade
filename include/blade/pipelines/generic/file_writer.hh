#ifndef BLADE_PIPELINES_GENERIC_FILE_WRITER_HH
#define BLADE_PIPELINES_GENERIC_FILE_WRITER_HH

#include <memory>
#include <deque>

#include "blade/pipeline.hh"

#include "blade/modules/guppi/writer.hh"

namespace Blade::Pipelines::Generic {

template<typename IT = CF32>
class BLADE_API FileWriter : public Pipeline {
 public:
    struct Config {
        std::string outputGuppiFile;
        bool directio;

        U64 stepNumberOfBeams;
        U64 stepNumberOfAntennas;
        U64 stepNumberOfFrequencyChannels;
        U64 stepNumberOfTimeSamples;
        U64 stepNumberOfPolarizations;

        U64 totalNumberOfFrequencyChannels;

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

    constexpr const U64 getStepNumberOfBeams() const {
        return guppi->getStepNumberOfBeams();
    }

    constexpr const U64 getStepNumberOfAntennas() const {
        return guppi->getStepNumberOfAntennas();
    }

    constexpr const U64 getStepNumberOfFrequencyChannels() const {
        return guppi->getStepNumberOfFrequencyChannels();
    }

    constexpr const U64 getStepNumberOfTimeSamples() const {
        return guppi->getStepNumberOfTimeSamples();
    }

    constexpr const U64 getStepNumberOfPolarizations() const {
        return guppi->getStepNumberOfPolarizations();
    }

    constexpr const U64 getStepInputBufferSize() const {
        return guppi->getStepInputBufferSize();
    }

    constexpr const U64 getTotalNumberOfBeams() const {
        return guppi->getTotalNumberOfBeams();
    }

    constexpr const U64 getTotalNumberOfAntennas() const {
        return guppi->getTotalNumberOfAntennas();
    }

    constexpr const U64 getTotalNumberOfFrequencyChannels() const {
        return guppi->getTotalNumberOfFrequencyChannels();
    }

    constexpr const U64 getTotalNumberOfTimeSamples() const {
        return guppi->getTotalNumberOfTimeSamples();
    }

    constexpr const U64 getTotalNumberOfPolarizations() const {
        return guppi->getTotalNumberOfPolarizations();
    }

    constexpr const U64 getTotalInputBufferSize() const {
        return guppi->getTotalInputBufferSize();
    }

    constexpr const U64 getNumberOfSteps() const {
        return guppi->getNumberOfSteps();
    }

    constexpr const Config& getConfig() const {
        return this->config;
    }

    const Result run();

 private:
    const Config config;

    Vector<Device::CPU, IT> writerBuffer;

    std::shared_ptr<Modules::Guppi::Writer<IT>> guppi;
};

}  // namespace Blade::Pipelines::Generic

#endif
