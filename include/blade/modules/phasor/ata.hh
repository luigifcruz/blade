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

    const Result preprocess(const cudaStream_t& stream, const U64& currentComputeCount) final;

 protected:
    const PhasorDimensions getOutputPhasorsDims() const {
        return {
            .B = this->config.beamCoordinates.size(),
            .A = this->config.numberOfAntennas,
            .F = this->config.numberOfFrequencyChannels,
            .T = 1,
            .P = this->config.numberOfPolarizations,
        };
    }

    const DelayDimensions getOutputDelaysDims() const {
        return {
            .B = this->config.beamCoordinates.size(),
            .A = this->config.numberOfAntennas,
        };
    }

    const ArrayDimensions getConfigCalibrationDims() const {
        return {
            .A = this->config.numberOfAntennas,
            .F = this->config.numberOfFrequencyChannels *
                 this->numberOfFrequencySteps,
            .T = 1,
            .P = this->config.numberOfPolarizations,
        };
    }

 private:
    std::vector<XYZ> antennasXyz;
    std::vector<UVW> boresightUvw;
    std::vector<UVW> sourceUvw;
    std::vector<F64> boresightDelay;
};

}  // namespace Blade::Modules::Phasor

#endif
