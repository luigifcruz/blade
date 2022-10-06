#ifndef BLADE_MODULES_PHASOR_VLA_HH
#define BLADE_MODULES_PHASOR_VLA_HH

#include "blade/modules/phasor/generic.hh"

namespace Blade::Modules::Phasor {

template<typename OT>
class BLADE_API VLA : public Module {
 public:
    // Configuration

    struct Config {
        U64 numberOfBeams;
        U64 numberOfAntennas;
        U64 numberOfFrequencyChannels;
        U64 numberOfPolarizations;

        F64 channelZeroFrequencyHz;
        F64 channelBandwidthHz;
        // F64 totalBandwidthHz;
        U64 frequencyStartIndex;

        // U64 referenceAntennaIndex;
        // LLA arrayReferencePosition; 
        // RA_DEC boresightCoordinate;
        // std::vector<XYZ> antennaPositions;
        ArrayTensor<Device::CPU, CF64> antennaCoefficients;
        PhasorTensor<Device::CPU, F64> beamAntennaDelays;
        Vector<Device::CPU, F64> delayTimes;
        // std::vector<RA_DEC> beamCoordinates;

        U64 preBeamformerChannelizerRate = 1;

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

    explicit VLA(const typename VLA<OT>::Config& config,
                 const typename VLA<OT>::Input& input);

    const Result preprocess(const cudaStream_t& stream = 0) final;

 protected:
    const PhasorTensorDimensions getOutputPhasorsDims() const {
        return {
            .B = this->config.numberOfBeams,
            .A = this->config.numberOfAntennas,
            .F = this->config.numberOfFrequencyChannels,
            .T = 1,
            .P = this->config.numberOfPolarizations,
        };
    }

    const DelayTensorDimensions getOutputDelaysDims() const {
        return {
            .B = this->config.numberOfBeams,
            .A = this->config.numberOfAntennas,
        };
    }

    const ArrayTensorDimensions getConfigCoefficientDims() const {
        return {
            .A = this->config.numberOfAntennas,
            .F = this->config.numberOfFrequencyChannels,
            .T = 1,
            .P = this->config.numberOfPolarizations,
        };
    }

 private:
    const Config config;
    const Input input;
    Output output;

    U64 frequencySteps;
    U64 frequencyStepIndex;
    U64 delayTimeIndex;
};

}  // namespace Blade::Modules::Phasor

#endif
