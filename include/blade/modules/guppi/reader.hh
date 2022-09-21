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
    // Configuration 

    struct Config {
        std::string filepath;
        U64 stepNumberOfTimeSamples;
        U64 stepNumberOfFrequencyChannels;
        U64 stepNumberOfAspects;

        U64 blockSize = 512;
    };

    constexpr const Config& getConfig() const {
        return this->config;
    }

    // Input 

    struct Input {
    };

    // Output

    struct Output {
        ArrayTensor<Device::CPU, OT> stepBuffer;
        Vector<Device::CPU, F64> stepJulianDate;
        Vector<Device::CPU, F64> stepDut1;
    };

    constexpr const ArrayTensor<Device::CPU, OT>& getStepOutputBuffer() const {
        return this->output.stepBuffer;
    }

    constexpr const Vector<Device::CPU, F64>& getStepOutputJulianDate() const {
        return this->output.stepJulianDate;
    }

    constexpr const Vector<Device::CPU, F64>& getStepOutputDut1() const {
        return this->output.stepDut1;
    }

    const ArrayTensorDimensions getTotalOutputBufferDims() const {
        return {
            .A = this->getTotalNumberOfAntennas(),
            .F = this->getTotalNumberOfFrequencyChannels(),
            .T = this->getTotalNumberOfTimeSamples(),
            .P = this->getTotalNumberOfPolarizations(),
        };
    }

    const U64 getNumberOfSteps() {
        return this->getTotalOutputBufferDims().size() / 
               this->getStepOutputBufferDims().size();
    }

    // Constructor & Processing

    explicit Reader(const Config& config, const Input& input);
    const Result preprocess(const cudaStream_t& stream = 0) final;

    // Dimension getters
    const U64 getStepNumberOfAntennas() const {
        return this->config.stepNumberOfAspects;
    }

    const U64 getStepNumberOfFrequencyChannels() const {
        return this->config.stepNumberOfFrequencyChannels;
    }

    const U64 getStepNumberOfTimeSamples() const {
        return this->config.stepNumberOfTimeSamples;
    }

    const U64 getStepNumberOfPolarizations() const {
        return this->getDatashape()->n_pol;
    }

    const U64 getStepOutputBufferSize() {
        return this->getStepOutputBufferDims().size();
    }

    const U64 getTotalNumberOfAntennas() const {
        return this->getDatashape()->n_aspect;
    }

    const U64 getTotalNumberOfFrequencyChannels() const {
        return this->getDatashape()->n_aspectchan;
    }

    const U64 getTotalNumberOfTimeSamples() const {
        return this->getDatashape()->n_time * this->gr_iterate.n_block;
    }

    const U64 getTotalNumberOfPolarizations() const {
        return this->getDatashape()->n_pol;
    }

    const U64 getTotalOutputBufferSize() {
        return this->getStepOutputBufferDims().size();
    }


    // Miscellaneous 

    const F64 getTotalBandwidth() const;
    const F64 getChannelBandwidth() const;
    const U64 getChannelStartIndex() const;
    const F64 getObservationFrequency() const;

 private:
    // Variables 

    Config config;
    const Input input;
    Output output;

    I32 lastread_block_index = -1;
    U64 lastread_aspect_index;
    U64 lastread_channel_index;
    U64 lastread_time_index;

    guppiraw_iterate_info_t gr_iterate = {0};

    // Expected Dimensions

    const ArrayTensorDimensions getStepOutputBufferDims() const {
        return {
            .A = this->getStepNumberOfAntennas(),
            .F = this->getStepNumberOfFrequencyChannels(),
            .T = this->getStepNumberOfTimeSamples(),
            .P = this->getStepNumberOfPolarizations(),
        };
    }

    // Helpers

    const bool keepRunning() const {
        return guppiraw_iterate_ntime_remaining(&this->gr_iterate) >=
            this->getStepNumberOfTimeSamples();
    }

    const guppiraw_datashape_t* getDatashape() const {
        return guppiraw_iterate_datashape(&this->gr_iterate);
    }
};

}  // namespace Blade::Modules

#endif
