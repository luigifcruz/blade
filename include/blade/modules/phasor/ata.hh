#ifndef BLADE_MODULES_PHASOR_ATA_HH
#define BLADE_MODULES_PHASOR_ATA_HH

#include "blade/modules/phasor/generic.hh"

namespace Blade::Modules::Phasor {

template<typename OT>
class BLADE_API ATA : public Generic<OT> {
 public:
    explicit ATA(const typename Generic<OT>::Config& config,
                 const typename Generic<OT>::Input& input,
                 const cudaStream_t& stream);

    Result process(const cudaStream_t& stream, const U64& currentStepNumber) final;

 protected:
    const PhasorShape getOutputPhasorsShape() const {
        return PhasorShape({
            this->config.beamCoordinates.size(),
            this->config.numberOfAntennas,
            this->config.numberOfFrequencyChannels,
            1,
            this->config.numberOfPolarizations,
        });
    }

    const DelayShape getOutputDelaysShape() const {
        return DelayShape({
            this->config.beamCoordinates.size(),
            this->config.numberOfAntennas,
        });
    }

    const ArrayShape getConfigCalibrationShape() const {
        return ArrayShape({
            this->config.numberOfAntennas,
            this->config.numberOfFrequencyChannels *
                 this->numberOfFrequencySteps,
            1,
            this->config.numberOfPolarizations,
        });
    }

 private:
    std::vector<XYZ> antennasXyz;
    std::vector<UVW> boresightUvw;
    std::vector<UVW> sourceUvw;
    std::vector<F64> boresightDelay;
};

}  // namespace Blade::Modules::Phasor

#endif
