#ifndef BLADE_MODULES_PHASOR_GENERIC_HH
#define BLADE_MODULES_PHASOR_GENERIC_HH

#include "blade/base.hh"
#include "blade/module.hh"

namespace Blade::Modules::Phasor {

template<typename OT>
class BLADE_API Generic : public Module {
 public:
    // Configuration

    struct Config {
        U64 numberOfAntennas;
        U64 numberOfFrequencyChannels;
        U64 numberOfPolarizations;

        F64 observationFrequencyHz;
        F64 channelBandwidthHz;
        F64 totalBandwidthHz;
        U64 frequencyStartIndex;

        U64 referenceAntennaIndex;
        LLA arrayReferencePosition; 
        RA_DEC boresightCoordinate;
        std::vector<XYZ> antennaPositions;
        std::vector<CF64> antennaCalibrations;
        std::vector<RA_DEC> beamCoordinates;

        U64 blockSize = 512;
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input 

    struct Input {
        const Vector<Device::CPU, F64>& blockJulianDate;
        const Vector<Device::CPU, F64>& blockDut1;
    };

    constexpr const Vector<Device::CPU, F64>& getInputJulianDate() const {
        return this->input.blockJulianDate;
    }

    constexpr const Vector<Device::CPU, F64>& getInputDut1() const {
        return this->input.blockDut1;
    }

    // Output

    struct Output {
        DelayTensor<Device::CPU, F64> delays;
        PhasorTensor<Device::CPU | Device::CUDA, OT> phasors;
    };

    constexpr const DelayTensor<Device::CPU, F64>& getOutputDelays() const {
        return this->output.delays;
    } 

    constexpr const PhasorTensor<Device::CPU | Device::CUDA, OT>& getOutputPhasors() const {
        return this->output.phasors;
    } 

    // Constructor & Processing

    explicit Generic(const Config& config, const Input& input);
    virtual ~Generic() = default;
    virtual const Result preprocess(const cudaStream_t& stream = 0) = 0;

 protected:
    // Variables

    const Config config;
    const Input input;
    Output output;

    // Expected Dimensions

    virtual const DelayTensorDimensions getOutputDelaysDims() const = 0;
    virtual const PhasorTensorDimensions getOutputPhasorsDims() const = 0;
    virtual const ArrayTensorDimensions getConfigCalibrationDims() const = 0;
};

}  // namespace Blade::Modules::Phasor

#endif
