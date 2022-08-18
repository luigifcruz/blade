#ifndef BLADE_MODULES_GUPPI_WRITER_HH
#define BLADE_MODULES_GUPPI_WRITER_HH

#include <filesystem>
#include <string>

#include "blade/base.hh"
#include "blade/module.hh"

extern "C" {
#include "guppirawc99.h"
}

namespace Blade::Modules::Guppi {

template<typename IT>
class BLADE_API Writer : public Module {
 public:
    struct Config {
        std::string filepath;
        bool directio;

        U64 stepNumberOfBeams;
        U64 stepNumberOfAntennas;
        U64 stepNumberOfFrequencyChannels;
        U64 stepNumberOfTimeSamples;
        U64 stepNumberOfPolarizations;

        U64 totalNumberOfFrequencyChannels;

        U64 blockSize = 512;
    };

    struct Input {
        const Vector<Device::CPU, IT>& totalBuffer;
    };

    struct Output {
    };

    explicit Writer(const Config& config, const Input& input);

    constexpr Vector<Device::CPU, IT>& getTotalInputBuffer() {
        return const_cast<Vector<Device::CPU, IT>&>(this->input.totalBuffer);
    }

    constexpr void headerPut(std::string key, std::string value) {
        guppiraw_header_put_string(&this->gr_header, key.c_str(), value.c_str());
    }

    constexpr void headerPut(std::string key, F64 value) {
        guppiraw_header_put_double(&this->gr_header, key.c_str(), value);
    }

    constexpr void headerPut(std::string key, I64 value) {
        guppiraw_header_put_integer(&this->gr_header, key.c_str(), value);
    }

    constexpr void headerPut(std::string key, I32 value) {
        guppiraw_header_put_integer(&this->gr_header, key.c_str(), (I64)value);
    }

    constexpr void headerPut(std::string key, U64 value) {
        guppiraw_header_put_integer(&this->gr_header, key.c_str(), (I64)value);
    }

    constexpr const Config& getConfig() const {
        return this->config;
    }

    constexpr const U64 getStepNumberOfBeams() const {
        return this->config.stepNumberOfBeams;
    }

    constexpr const U64 getStepNumberOfAntennas() const {
        return this->config.stepNumberOfAntennas;
    }

    constexpr const U64 getStepNumberOfFrequencyChannels() const {
        return this->config.stepNumberOfFrequencyChannels;
    }

    constexpr const U64 getStepNumberOfTimeSamples() const {
        return this->config.stepNumberOfTimeSamples;
    }

    constexpr const U64 getStepNumberOfPolarizations() const {
        return this->config.stepNumberOfPolarizations;
    }

    constexpr const U64 getStepInputBufferSize() const {
        return this->getNumberOfAspects() *
               this->getStepNumberOfFrequencyChannels() *
               this->getStepNumberOfTimeSamples() * 
               this->getStepNumberOfPolarizations();
    }

    constexpr const U64 getTotalNumberOfBeams() const {
        return this->getStepNumberOfBeams();
    }

    constexpr const U64 getTotalNumberOfAntennas() const {
        return this->getStepNumberOfAntennas();
    }

    constexpr const U64 getTotalNumberOfFrequencyChannels() const {
        return this->config.totalNumberOfFrequencyChannels;
    }

    constexpr const U64 getTotalNumberOfTimeSamples() const {
        return this->getStepNumberOfTimeSamples();
    }

    constexpr const U64 getTotalNumberOfPolarizations() const {
        return this->getStepNumberOfPolarizations();
    }

    constexpr const U64 getTotalInputBufferSize() const {
        return this->getNumberOfAspects() *
               this->getTotalNumberOfFrequencyChannels() *
               this->getTotalNumberOfTimeSamples() * 
               this->getTotalNumberOfPolarizations();
    }

    constexpr const U64 getNumberOfSteps() const {
        return this->getTotalNumberOfFrequencyChannels() / this->getStepNumberOfFrequencyChannels();
    }

    const Result preprocess(const cudaStream_t& stream = 0) final;

 private:
    Config config;
    Input input;
    Output output;

    U64 fileId;
    U64 writeCounter;
    I32 fileDescriptor;

    guppiraw_header_t gr_header = {0};

    // TODO: Behavior unclear. Zeroed numberOfBeams is undefined. Do we need this?
    constexpr const U64 getNumberOfAspects() const {
        return this->config.stepNumberOfBeams > 0 ? 
            this->config.stepNumberOfBeams : this->config.stepNumberOfAntennas;
    }
};

}  // namespace Blade::Modules

#endif
