#ifndef BLADE_MODULES_SETICORE_HITS_RAW_WRITER_HH
#define BLADE_MODULES_SETICORE_HITS_RAW_WRITER_HH

#include <filesystem>
#include <string>

#include "blade/base.hh"
#include "blade/module.hh"

extern "C" {
#include "guppirawc99.h"
}

#include "dedoppler.h"
#include "dedoppler_hit_group.h"

namespace Blade::Modules::Seticore {

template<typename IT>
class BLADE_API HitsRawWriter : public Module {
 public:
    // Configuration 

    struct Config {
        std::string filepathPrefix;
        bool directio = true;

        U64 telescopeId;
        std::string sourceName;
        std::string observationIdentifier;
        RA_DEC phaseCenter;
        U64 coarseStartChannelIndex;
        U64 coarseChannelRatio;
        F64 channelBandwidthHz;
        F64 channelTimespanS;
        F64 stampFrequencyMarginHz = 500.0;
        I64 hitsGroupingMargin = 30;

        U64 blockSize = 512;
    };

    constexpr const Config& getConfig() const {
        return this->config;
    }

    // Input

    struct Input {
        const ArrayTensor<Device::CPU, IT>& buffer;
        std::vector<DedopplerHit>& hits;
        const Vector<Device::CPU, F64>& frequencyOfFirstChannelHz;
        const Vector<Device::CPU, F64>& julianDateStart;
    };

    constexpr const ArrayTensor<Device::CPU, IT>& getInputBuffer() const {
        return this->input.buffer;
    }

    constexpr const Vector<Device::CPU, F64>& getInputFrequencyOfFirstChannelHz() const {
        return this->input.frequencyOfFirstChannelHz;
    }

    constexpr const Vector<Device::CPU, F64>& getInputJulianDateStart() const {
        return this->input.julianDateStart;
    }

    // Output

    struct Output {
    };

    // Constructor & Processing

    explicit HitsRawWriter(const Config& config, const Input& input);
    const Result process(const cudaStream_t& stream = 0) final;

    // Miscellaneous

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
    I32 fileDescriptor;

    guppiraw_header_t gr_header = {0};
};

}  // namespace Blade::Modules::Seticore

#endif
