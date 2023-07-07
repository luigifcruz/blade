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
    // Configuration 

    struct Config {
        std::string filepath;
        bool directio = true;

        U64 inputFrequencyBatches;

        U64 blockSize = 512;
    };

    constexpr const Config& getConfig() const {
        return this->config;
    }

    // Input

    struct Input {
        const ArrayTensor<Device::CPU, IT>& buffer;
    };

    constexpr const ArrayTensor<Device::CPU, IT>& getInputBuffer() const {
        return this->input.buffer;
    }

    // Output

    struct Output {
    };

    // Taint Registers

    constexpr const MemoryTaint getMemoryTaint() {
        return MemoryTaint::CONSUMER; 
    }

    // Constructor & Processing

    explicit Writer(const Config& config, const Input& input,
                    const cudaStream_t& stream);
    Result preprocess(const cudaStream_t& stream, const U64& currentComputeCount) final;

    // Miscullaneous

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

 private:
    // Variables

    Config config;
    Input input;
    Output output;

    U64 fileId;
    U64 writeCounter;
    I32 fileDescriptor;

    guppiraw_header_t gr_header = {0};
};

}  // namespace Blade::Modules

#endif
