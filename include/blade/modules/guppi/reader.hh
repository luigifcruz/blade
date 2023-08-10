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
        U64 requiredMultipleOfTimeSamplesSteps = 1;
        U64 stepNumberOfFrequencyChannels;

        U64 numberOfTimeSampleStepsBeforeFrequencyChannelStep = 1;
        U64 blockSize = 512;
        U64 numberOfFilesLimit = 0; // zero for no limit
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
        Vector<Device::CPU, U64> stepFrequencyChannelOffset;
    };

    constexpr const ArrayTensor<Device::CPU, OT>& getStepOutputBuffer() const {
        return this->output.stepBuffer;
    }

    F64 getUnixDateOfLastReadBlock(const U64 timesamplesOffset = 0);

    constexpr F64 getJulianDateOfLastReadBlock(const U64 timesamplesOffset = 0) {
        return calc_julian_date_from_unix_sec(this->getUnixDateOfLastReadBlock(timesamplesOffset));
    }

    constexpr const Vector<Device::CPU, F64>& getStepOutputJulianDate() const {
        return this->output.stepJulianDate;
    }

    constexpr const Vector<Device::CPU, F64>& getStepOutputDut1() const {
        return this->output.stepDut1;
    }

    constexpr const Vector<Device::CPU, U64>& getStepOutputFrequencyChannelOffset() const {
        return this->output.stepFrequencyChannelOffset;
    }

    const ArrayDimensions getTotalOutputBufferDims() const {
        return {
            .A = this->getDatashape()->n_aspect,
            .F = this->getDatashape()->n_aspectchan,
            .T = this->getDatashape()->n_time * this->gr_iterate.n_block,
            .P = this->getDatashape()->n_pol,
        };
    }

    const ArrayDimensions getStepOutputBufferDims() const {
        return {
            .A = this->getDatashape()->n_aspect,
            .F = this->config.stepNumberOfFrequencyChannels,
            .T = this->config.stepNumberOfTimeSamples,
            .P = this->getDatashape()->n_pol,
        };
    }

    const ArrayDimensions getNumberOfStepsInDimensions() const {
        auto dimensionSteps = this->getTotalOutputBufferDims() / this->getStepOutputBufferDims();
        if (this->config.numberOfTimeSampleStepsBeforeFrequencyChannelStep > 0) {
            dimensionSteps.T -= dimensionSteps.T % this->config.numberOfTimeSampleStepsBeforeFrequencyChannelStep;
        }
        dimensionSteps.T -= dimensionSteps.T % this->config.requiredMultipleOfTimeSamplesSteps;
        return dimensionSteps;
    }

    const U64 getNumberOfSteps() {
        return this->getNumberOfStepsInDimensions().size();
    }

    // Constructor & Processing

    explicit Reader(const Config& config, const Input& input);
    const Result preprocess(const cudaStream_t& stream, const U64& currentComputeCount) final;

    // Miscellaneous 

    const F64 getObservationBandwidth() const;
    const F64 getChannelBandwidth() const;
    const F64 getChannelTimespan() const;
    const U64 getChannelStartIndex() const;
    const F64 getObservationCenterFrequency() const;
    const F64 getCenterFrequency() const;
    const F64 getObservationBottomFrequency() const;
    const F64 getBottomFrequency() const;
    const F64 getObservationTopFrequency() const;
    const F64 getTopFrequency() const;
    const F64 getBandwidth() const;
    const F64 getAzimuthAngle() const;
    const F64 getZenithAngle() const;
    const F64 getRightAscension() const;
    const F64 getDeclination() const;
    const F64 getPhaseRightAscension() const;
    const F64 getPhaseDeclination() const;
    const std::string getSourceName() const;
    const std::string getTelescopeName() const;

 private:
    // Variables 

    Config config;
    const Input input;
    Output output;

    U64 lastread_channel_index = 0;
    U64 lastread_block_index = 0;
    U64 lastread_time_index = 0;

    U64 current_time_sample_step = 0;
    U64 checkpoint_block_index = 0;
    U64 checkpoint_time_index = 0;

    guppiraw_iterate_info_t gr_iterate = {0};

    // Helpers

    const bool keepRunning() const {
        // const auto numberOfStepsInDimensions = this->getNumberOfStepsInDimensions();
        // return this->currentStepDimensionIndices.T < numberOfStepsInDimensions.T &&
        // this->currentStepDimensionIndices.F < numberOfStepsInDimensions.F;
        return guppiraw_iterate_ntime_remaining(&this->gr_iterate) >= 
                this->config.stepNumberOfTimeSamples;
    }

    const guppiraw_datashape_t* getDatashape() const {
        return guppiraw_iterate_datashape(&this->gr_iterate);
    }
};

}  // namespace Blade::Modules

#endif
