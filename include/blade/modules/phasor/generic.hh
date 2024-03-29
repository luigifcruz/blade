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
        ArrayTensor<Device::CPU, CF64> antennaCalibrations;
        std::vector<RA_DEC> beamCoordinates;

        U64 blockSize = 512;
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input 

    struct Input {
        const Tensor<Device::CPU, F64>& blockJulianDate;
        const Tensor<Device::CPU, F64>& blockDut1;
    };

    constexpr const Tensor<Device::CPU, F64>& getInputJulianDate() const {
        return this->input.blockJulianDate;
    }

    constexpr const Tensor<Device::CPU, F64>& getInputDut1() const {
        return this->input.blockDut1;
    }

    // Output

    struct Output {
        DelayTensor<Device::CPU, F64> delays;
        PhasorTensor<Device::CUDA, OT> phasors;
    };

    constexpr const DelayTensor<Device::CPU, F64>& getOutputDelays() const {
        return this->output.delays;
    } 

    constexpr const PhasorTensor<Device::CUDA, OT>& getOutputPhasors() const {
        return this->output.phasors;
    } 

    // Taint Registers

    constexpr Taint getTaint() const {
        return Taint::CONSUMER |
               Taint::PRODUCER; 
    }

    std::string name() const {
        return "Phasor";
    }

    // Constructor & Processing

    explicit Generic(const Config& config, const Input& input, const Stream& stream = {});
    virtual ~Generic() = default;
    virtual Result process(const U64& currentComputeCount, const Stream& stream = {}) = 0;

 protected:
    // Variables

    const Config config;
    const Input input;
    Output output;

    // Expected Shape

    virtual const DelayShape getOutputDelaysShape() const = 0;
    virtual const PhasorShape getOutputPhasorsShape() const = 0;
    virtual const ArrayShape getConfigCalibrationShape() const = 0;

    // Miscellaneous

    U64 numberOfFrequencySteps;
    U64 currentFrequencyStep;
};

}  // namespace Blade::Modules::Phasor

#endif
