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

        LLA arrayReferencePosition; 
        HA_DEC boresightCoordinate;

        std::vector<XYZ> antennaPositions;
        std::vector<F64> antennaCalibrations; 
        std::vector<RA_DEC> beamCoordinates;

        U64 blockSize = 512;
    };

    struct Input {
    };

    struct Output {
        Vector<Device::CUDA, OT> phasors;
    };

    explicit Generic(const Config& config, const Input& input);
    virtual ~Generic() = default;

    constexpr Vector<Device::CUDA, OT>& getPhasors() {
        return this->output.phasors;
    } 

    constexpr Config getConfig() const {
        return config;
    }

    virtual constexpr U64 getPhasorsSize() const = 0;

    Result preprocess(const cudaStream_t& stream = 0) final;
    Result process(const cudaStream_t& stream = 0) final;

 protected:
    const Config config;
    const Input input;
    Output output;
};

}  // namespace Blade::Modules::Phasor

#endif
