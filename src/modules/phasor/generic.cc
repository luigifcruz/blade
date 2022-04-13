#include "blade/modules/phasor/generic.hh"

#include "phasor.jit.hh"

namespace Blade::Modules::Phasor {

template<typename OT>
Generic<OT>::Generic(const Config& config, const Input& input)
        : Module(config.blockSize, phasor_kernel),
          config(config),
          input(input) {
    if (config.numberOfBeams != config.beamCoordinates.size()) {
        BL_FATAL("Number of Beams configuration ({}) mismatches the number of"
                 " beams coordinates ({}).", config.numberOfBeams,
                 config.beamCoordinates.size());
        throw Result::ERROR;
    }

    if (config.numberOfAntennas != config.antennaPositions.size()) {
        BL_FATAL("Number of Antennas configuration ({}) mismatches the number of"
                 " antenna positions ({}).", config.numberOfAntennas,
                 config.antennaPositions.size());
        throw Result::ERROR;
    }

    if (config.numberOfAntennas * config.numberOfFrequencyChannels * config.numberOfPolarizations
            != config.antennaCalibrations.size()) {
        BL_FATAL("Insufficient number of antenna calibrations ({}). This number"
                 " should be the product of Number of Antennas ({}) and Number of"
                 " Frequency Channels ({}).", config.antennaCalibrations.size(), 
                 config.numberOfAntennas, config.numberOfFrequencyChannels);
        throw Result::ERROR;
    }
    
    if (config.referenceAntennaIndex >= config.numberOfAntennas) {
        BL_FATAL("Reference Antenna Index ({}) is larger than the number of"
                 " antennas ({}).", config.referenceAntennaIndex,
                 config.numberOfAntennas);
        throw Result::ERROR;
    }
}

template<typename OT>
Result Generic<OT>::preprocess(const cudaStream_t& stream) {
    return Result::SUCCESS;
}

template<typename OT>
Result Generic<OT>::process(const cudaStream_t& stream) {
    return Result::SUCCESS;
}

template class BLADE_API Generic<CF32>;
template class BLADE_API Generic<CF64>;

}  // namespace Blade::Modules::Phasor
