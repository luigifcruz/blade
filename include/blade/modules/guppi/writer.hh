#ifndef BLADE_MODULES_GUPPI_WRITER_HH
#define BLADE_MODULES_GUPPI_WRITER_HH

#include <filesystem>
#include <string>

#include "blade/base.hh"
#include "blade/module.hh"

extern "C" {
#include "guppiraw.h"
}

namespace Blade::Modules::Guppi {

template<typename IT>
class BLADE_API Writer : public Module {
 public:
    struct Config {
        std::string filepath;
        bool directio;

        U64 numberOfBeams;
        U64 numberOfAntennas;
        U64 numberOfFrequencyChannels;
        U64 numberOfTimeSamples;
        U64 numberOfPolarizations;

        U64 totalNumberOfFrequencyChannels;

        U64 blockSize = 512;
    };

    struct Input {
        Vector<Device::CPU, IT> buf;
    };

    struct Output {
    };

    explicit Writer(const Config& config);

    constexpr const U64 getInputBatchOffset(U64 channel_batch_index) {
        return channel_batch_index * this->getInputSize();
    }

    constexpr Vector<Device::CPU, IT>& getInput() {
        return this->input.buf;
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

    constexpr const U64 getNumberOfBeams() const {
        return this->config.numberOfBeams;
    }

    constexpr const U64 getNumberOfAntennas() const {
        return this->config.numberOfAntennas;
    }

    constexpr const U64 getNumberOfFrequencyChannels() const {
        return this->config.numberOfFrequencyChannels;
    }

    constexpr const U64 getNumberOfTimeSamples() const {
        return this->config.numberOfTimeSamples;
    }

    constexpr const U64 getNumberOfPolarizations() const {
        return this->config.numberOfPolarizations;
    }

    constexpr const U64 getInputSize() const {
        return this->getNumberOfAspects() *
               this->getNumberOfFrequencyChannels() *
               this->getNumberOfTimeSamples() * 
               this->getNumberOfPolarizations();
    }

    constexpr const U64 getTotalNumberOfFrequencyChannels() const {
        return this->config.totalNumberOfFrequencyChannels;
    }

    constexpr const U64 getTotalInputSize() const {
        return this->getNumberOfAspects() *
               this->getTotalNumberOfFrequencyChannels() *
               this->getNumberOfTimeSamples() * 
               this->getNumberOfPolarizations();
    }

    constexpr const U64 getNumberOfBatches() const {
        return this->getTotalNumberOfFrequencyChannels() / this->getNumberOfFrequencyChannels();
    }

    Result preprocess(const cudaStream_t& stream = 0) final;

 private:
    Config config;
    Input input;
    Output output;

    U64 file_id = 0;
    I32 file_descriptor = 0;
    U64 fileblock_index = 0;
    Vector<Device::CPU, IT> writeBuffer;

    guppiraw_header_t gr_header = {0};

    // TODO: Behavior unclear. Zeroed numberOfBeams is undefined. Do we need this?
    constexpr const U64 getNumberOfAspects() const {
        return this->config.numberOfBeams > 0 ? 
            this->config.numberOfBeams : this->config.numberOfAntennas;
    }
};

}  // namespace Blade::Modules

#endif
