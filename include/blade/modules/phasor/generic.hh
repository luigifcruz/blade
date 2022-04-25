#ifndef BLADE_MODULES_PHASOR_GENERIC_HH
#define BLADE_MODULES_PHASOR_GENERIC_HH

#include "blade/base.hh"
#include "blade/module.hh"

namespace Blade::Modules::Phasor {

template<typename OT>
class BLADE_API Generic : public Module {
 public:
    struct Config {
        U64 numberOfBeams;
        U64 numberOfAntennas;
        U64 numberOfFrequencyChannels;
        U64 numberOfPolarizations;

        F64 rfFrequencyHz;
        F64 channelBandwidthHz;
        F64 totalBandwidthHz;
        U64 frequencyStartIndex;

        U64 referenceAntennaIndex;
        LLA arrayReferencePosition; 
        RA_DEC boresightCoordinate;
        std::vector<XYZ> antennaPositions;
        std::vector<F64> antennaCalibrations; 
        std::vector<RA_DEC> beamCoordinates;

        U64 blockSize = 512;
    };

    struct Input {
        const F64& frameJulianDate;
        const F64& frameDut1;
    };

    struct Output {
        Vector<Device::CPU, F64> delays;
        Vector<Device::CPU | Device::CUDA, OT> phasors;
    };

    explicit Generic(const Config& config, const Input& input);
    virtual ~Generic() = default;

    constexpr Vector<Device::CPU, F64>& getDelays() {
        return this->output.delays;
    } 

    constexpr Vector<Device::CPU | Device::CUDA, OT>& getPhasors() {
        return this->output.phasors;
    } 

    constexpr Config getConfig() const {
        return config;
    }

    virtual constexpr U64 getPhasorsSize() const = 0;
    virtual constexpr U64 getDelaysSize() const = 0;
    virtual constexpr U64 getCalibrationsSize() const = 0;

    virtual Result preprocess(const cudaStream_t& stream = 0) = 0;

 protected:
    const Config config;
    const Input input;
    Output output;
};

}  // namespace Blade::Modules::Phasor

#endif
