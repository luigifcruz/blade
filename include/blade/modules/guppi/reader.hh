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
        Tensor<Device::CPU, F64> stepJulianDate;
        Tensor<Device::CPU, F64> stepDut1;
    };

    constexpr const ArrayTensor<Device::CPU, OT>& getStepOutputBuffer() const {
        return this->output.stepBuffer;
    }

    constexpr const Tensor<Device::CPU, F64>& getStepOutputJulianDate() const {
        return this->output.stepJulianDate;
    }

    constexpr const Tensor<Device::CPU, F64>& getStepOutputDut1() const {
        return this->output.stepDut1;
    }

    const ArrayShape getTotalOutputBufferShape() const {
        return ArrayShape({
            this->getDatashape()->n_aspect,
            this->getDatashape()->n_aspectchan,
            this->getDatashape()->n_time * this->gr_iterate.n_block,
            this->getDatashape()->n_pol,
        });
    }

    const ArrayShape getStepOutputBufferShape() const {
        return ArrayShape({
            this->config.stepNumberOfAspects,
            this->config.stepNumberOfFrequencyChannels,
            this->config.stepNumberOfTimeSamples,
            this->getDatashape()->n_pol,
        });
    }

    const U64 getNumberOfSteps() {
        return this->getTotalOutputBufferShape().size() / 
               this->getStepOutputBufferShape().size();
    }

    // Taint Registers

    constexpr Taint getTaint() const {
        return Taint::PRODUCER; 
    }

    // Constructor & Processing

    explicit Reader(const Config& config, const Input& input, 
                    const cudaStream_t& stream);
    Result process(const cudaStream_t& stream, const U64& currentStepNumber) final;

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

    // Helpers

    const bool keepRunning() const {
        return guppiraw_iterate_ntime_remaining(&this->gr_iterate) >= 
                this->config.stepNumberOfTimeSamples;
    }

    const guppiraw_datashape_t* getDatashape() const {
        return guppiraw_iterate_datashape(&this->gr_iterate);
    }
};

}  // namespace Blade::Modules

#endif
