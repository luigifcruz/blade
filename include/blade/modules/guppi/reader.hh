#ifndef BLADE_MODULES_GUPPI_READER_HH
#define BLADE_MODULES_GUPPI_READER_HH

#include <filesystem>
#include <string>

#include "blade/base.hh"
#include "blade/module.hh"

extern "C" {
#include "guppirawc99.h"
#include "radiointerferometryc99.h"
}

namespace Blade::Modules::Guppi {

template<typename OT>
class BLADE_API Reader : public Module {
 public:
    struct Config {
        std::string filepath;

        U64 stepNumberOfTimeSamples;
        U64 stepNumberOfFrequencyChannels;
        U64 stepNumberOfAntennas;

        U64 blockSize = 512;
    };

    struct Input {
    };

    struct Output {
        Vector<Device::CPU, OT> stepBuffer;
        Vector<Device::CPU, F64> stepJulianDate;
        Vector<Device::CPU, F64> stepDut1;
    };

    explicit Reader(const Config& config, const Input& input);

    constexpr const bool keepRunning() const {
        return guppiraw_iterate_ntime_remaining(&this->gr_iterate) > this->getStepNumberOfTimeSamples();
    }

    constexpr const Vector<Device::CPU, OT>& getStepOutputBuffer() {
        return this->output.stepBuffer;
    }

    constexpr const Vector<Device::CPU, F64>& getStepOutputJulianDate() {
        return this->output.stepJulianDate;
    }

    constexpr const Vector<Device::CPU, F64>& getStepOutputDut1() {
        return this->output.stepDut1;
    }

    constexpr const Config& getConfig() const {
        return this->config;
    }

    constexpr const U64 getLastReadBlockIndex() const {
        return this->lastread_block_index;
    }

    constexpr const U64 getLastReadAspectIndex() const {
        return this->lastread_aspect_index;
    }

    constexpr const U64 getLastReadChannelIndex() const {
        return this->lastread_channel_index;
    }

    constexpr const U64 getLastReadTimeIndex() const {
        return this->lastread_time_index;
    }

    const F64 getTotalBandwidth();
    const F64 getChannelBandwidth();
    const U64 getChannelStartIndex();
    const F64 getObservationFrequency();

    constexpr const U64 getTotalNumberOfAntennas() const {
        return this->getDatashape()->n_aspect;
    }

    constexpr const U64 getTotalNumberOfFrequencyChannels() const {
        return this->getDatashape()->n_aspectchan;
    }

    constexpr const U64 getTotalNumberOfPolarizations() const {
        return this->getDatashape()->n_pol;
    }

    constexpr const U64 getTotalNumberOfTimeSamples() const {
        return this->getDatashape()->n_time * this->gr_iterate.n_block;
    }

    constexpr const U64 getTotalOutputBufferSize() const {
        return this->getTotalNumberOfAntennas() *
               this->getTotalNumberOfFrequencyChannels() *
               this->getTotalNumberOfTimeSamples() * 
               this->getTotalNumberOfPolarizations();
    }

    constexpr const U64 getStepNumberOfAntennas() const {
        return this->config.stepNumberOfAntennas;
    }

    constexpr const U64 getStepNumberOfFrequencyChannels() const {
        return this->config.stepNumberOfFrequencyChannels;
    }

    constexpr const U64 getStepNumberOfPolarizations() const {
        return this->getTotalNumberOfPolarizations();
    }

    constexpr const U64 getStepNumberOfTimeSamples() const {
        return this->config.stepNumberOfTimeSamples;
    }

    constexpr const U64 getStepOutputBufferSize() const {
        return this->getStepNumberOfAntennas() *
               this->getStepNumberOfFrequencyChannels() *
               this->getStepNumberOfTimeSamples() * 
               this->getStepNumberOfPolarizations();
    }

    constexpr const U64 getBlockNumberOfTimeSamples() {
        return this->getDatashape()->n_time;
    }

    // TODO: Add getNumberOfSteps() method.

    const Result preprocess(const cudaStream_t& stream = 0) final;

 private:
    Config config;
    const Input input;
    Output output;

    I64 lastread_block_index = -1;
    U64 lastread_aspect_index;
    U64 lastread_channel_index;
    U64 lastread_time_index;

    guppiraw_iterate_info_t gr_iterate = {0};

    constexpr const guppiraw_datashape_t* getDatashape() const {
        return guppiraw_iterate_datashape(&this->gr_iterate);
    }
};

}  // namespace Blade::Modules

#endif
