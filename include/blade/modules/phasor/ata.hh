#ifndef BLADE_MODULES_PHASOR_ATA_HH
#define BLADE_MODULES_PHASOR_ATA_HH

#include "blade/modules/phasor/generic.hh"

namespace Blade::Modules::Phasor {

template<typename OT>
class BLADE_API ATA : public Generic<OT> {
 public:
    explicit ATA(const typename Generic<OT>::Config& config,
                 const typename Generic<OT>::Input& input);

    const Result preprocess(const cudaStream_t& stream = 0) final;

 protected:
    constexpr const PhasorTensorDimensions getOutputPhasorsDims() const {
        return {
            this->config.numberOfBeams,
            this->config.numberOfAntennas,
            this->config.numberOfFrequencyChannels,
            this->config.numberOfPolarizations,
        };
    }

    constexpr const DelayTensorDimensions getOutputDelaysDims() const {
        return {
            this->config.numberOfBeams,
            this->config.numberOfAntennas,
        };
    }

    constexpr const ArrayTensorDimensions getConfigCalibrationDims() const {
        return {
            this->config.numberOfAntennas,
            this->config.numberOfFrequencyChannels,
            this->config.numberOfPolarizations,
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
