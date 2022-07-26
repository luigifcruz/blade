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
        std::string filepathStem;

        bool directio;

        U64 numberOfAntennas;
        U64 numberOfBeams;
        U64 numberOfFrequencyChannels;
        U64 totalNumberOfFrequencyChannels;
        U64 numberOfTimeSamples;
        U64 numberOfPolarizations;

        U64 blockSize = 512;
    };

    struct Input {
        Vector<Device::CPU, IT> buf;
    };

    struct Output {
    };

    explicit Writer(const Config& config);

    constexpr const U64 write() const {
        return guppiraw_write_block_batched(this->file_descriptor, &this->gr_header, this->input.buf.data(), 1, this->getNumberOfFrequencyChannelBatches());
    }

    constexpr const U64 getInputBatchOffset(U64 channel_batch_index) {
        return channel_batch_index * this->getInputBatchSize();
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

    constexpr const U64 getNumberOfAspects() const {
        return (this->config.numberOfBeams > 0 ? this->config.numberOfBeams : this->config.numberOfAntennas);
    }

    constexpr const U64 getNumberOfFrequencyChannels() const {
        return this->config.numberOfFrequencyChannels;
    }

    constexpr const U64 getTotalNumberOfFrequencyChannels() const {
        return this->config.totalNumberOfFrequencyChannels;
    }

    constexpr const U64 getNumberOfFrequencyChannelBatches() const {
        return this->getTotalNumberOfFrequencyChannels() / this->getNumberOfFrequencyChannels();
    }

    constexpr const U64 getNumberOfPolarizations() const {
        return this->config.numberOfPolarizations;
    }

    constexpr const U64 getNumberOfTimeSamples() const {
        return this->config.numberOfTimeSamples;
    }

    constexpr const U64 getInputSize() const {
        return this->getNumberOfAspects() *
               this->getTotalNumberOfFrequencyChannels() *
               this->getNumberOfTimeSamples() * 
               this->getNumberOfPolarizations();
    }

    constexpr const U64 getInputBatchSize() const {
        return this->getNumberOfAspects() *
               this->getNumberOfFrequencyChannels() *
               this->getNumberOfTimeSamples() * 
               this->getNumberOfPolarizations();
    }

 private:
    Config config;
    Input input;
    Output output;

    int file_descriptor = 0;
    std::string filepath;
    U64 file_id = 0;
    U64 fileblock_index = 0;

    guppiraw_header_t gr_header = {0};

};

}  // namespace Blade::Modules

#endif