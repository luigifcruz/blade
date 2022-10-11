#define BL_LOG_DOMAIN "M::PHASOR::VLA"

#include "blade/modules/phasor/vla.hh"
#include "phasor.jit.hh"

extern "C" {
#include "radiointerferometryc99.h"
}

namespace Blade::Modules::Phasor {

template<typename OT>
VLA<OT>::VLA(const typename VLA<OT>::Config& config,
             const typename VLA<OT>::Input& input)
    : Module(config.blockSize, phasor_kernel),
      config(config),
      input(input) {
    // Check configuration values.
    const auto coefficientCoarseStepDims = this->getConfigCoefficientDims() / ArrayTensorDimensions({
        .A = 1,
        .F = this->config.preBeamformerChannelizerRate,
        .T = 1,
        .P = 1,
    });
    if (config.antennaCoefficients.size() % coefficientCoarseStepDims.size() != 0) {
        BL_FATAL("Number of antenna coefficients ({}) is not the expected size ({}), nor an integer multiple (on the frequency axis).", 
                config.antennaCoefficients.size(), coefficientCoarseStepDims);
        BL_CHECK_THROW(Result::ERROR);
    }

    this->frequencySteps = config.antennaCoefficients.size() / coefficientCoarseStepDims.size();
    this->frequencyStepIndex = 0;

    const auto coefficientTotalDims =
        coefficientCoarseStepDims * ArrayTensorDimensions({
            .A = 1,
            .F = this->frequencySteps,
            .T = 1,
            .P = 1,
        });
    this->antennaCoefficients.resize(coefficientTotalDims);
    BL_CHECK_THROW(Memory::Copy(this->antennaCoefficients, config.antennaCoefficients));

    this->beamAntennaDelays.resize({
        .B = this->getNumberOfBeams(),
        .A = config.numberOfAntennas,
        .F = 1,
        .T = 1,
        .P = 1,
    });
    BL_CHECK_THROW(Memory::Copy(this->beamAntennaDelays, beamAntennaDelays));
    
    this->delayTimes.resize({config.delayTimes.size()});
    BL_CHECK_THROW(Memory::Copy(this->delayTimes, config.delayTimes));

    this->delayTimeIndex = 0;

    // Allocate output buffers.
    BL_CHECK_THROW(this->output.phasors.resize(getOutputPhasorsDims()));
    BL_CHECK_THROW(this->output.delays.resize(getOutputDelaysDims()));

    // Print configuration values.
    BL_INFO("Coefficient Dimensions [A, F, T, P]: {}", coefficientTotalDims);
    BL_INFO("Delays Dimensions [B, A]: {} -> {}", this->getOutputDelays().dims(), "N/A");
    BL_INFO("Phasors Dimensions [B, A, F, T, P]: {} -> {}", "N/A", this->getOutputPhasors().dims());
    BL_INFO("Frequency steps: {}", this->frequencySteps);
    BL_INFO("Delay Times:")
    time_t curtime = 0;
    char timestr[32] = {0};
    for (U64 i = 0; i < this->delayTimes.size(); i++) {
        curtime = this->delayTimes[i];
        ctime_r(&curtime, timestr);
        timestr[strlen(timestr)-1] = '\0'; // Chop off trailing newline
        BL_INFO("   {}: {} ({})", i, this->delayTimes[i], timestr);
    }
}

template<typename OT>
const Result VLA<OT>::preprocess(const cudaStream_t& stream) {
    if (this->delayTimeIndex < this->delayTimes.size()-1) {
        const auto delayTimeThreshold = (3*this->delayTimes[this->delayTimeIndex+1] - this->delayTimes[this->delayTimeIndex]) / 2.0;
        const auto blockUnixS = calc_unix_sec_from_julian_date(this->input.blockJulianDate[0]);
        if (blockUnixS >= delayTimeThreshold) {
            this->delayTimeIndex += 1;
        }
    }

    const U64 currentFrequencyStepOffset = this->frequencyStepIndex * this->config.numberOfFrequencyChannels;
    const auto beamAntennaDelayDims = this->beamAntennaDelays.dims();
    const U64 currentDelayTimeOffset = this->delayTimeIndex * beamAntennaDelayDims.size()/beamAntennaDelayDims.numberOfTimeSamples();
    const U64 delayBeamStride = beamAntennaDelayDims.size()/(beamAntennaDelayDims.numberOfTimeSamples()*beamAntennaDelayDims.numberOfBeams());

    for (U64 b = 0; b < this->getNumberOfBeams(); b++) {
        const U64 beamOffset = (b *
                                this->config.numberOfAntennas *
                                this->config.numberOfFrequencyChannels *
                                this->config.numberOfPolarizations);

        for (U64 a = 0; a < this->config.numberOfAntennas; a++) {
            const U64 antennaPhasorOffset = (a *
                                       this->config.numberOfFrequencyChannels *
                                       this->config.numberOfPolarizations);
            const U64 antennaCoeffOffset = antennaPhasorOffset / this->config.preBeamformerChannelizerRate;

            const F64 delay = this->beamAntennaDelays[currentDelayTimeOffset + b*delayBeamStride + a];

            for (U64 f = 0; f < this->config.numberOfFrequencyChannels; f++) {
                const U64 frequencyPhasorOffset = (f * this->config.numberOfPolarizations);
                const U64 frequencyCoeffOffset = ((f + currentFrequencyStepOffset) / this->config.preBeamformerChannelizerRate) * this->config.numberOfPolarizations;

                const F64 freq = this->config.channelZeroFrequencyHz + this->config.frequencyStartIndex * this->config.channelBandwidthHz + (f + currentFrequencyStepOffset) * this->config.channelBandwidthHz / this->config.preBeamformerChannelizerRate;
                const CF64 phasorsExp(0, -2 * BL_PHYSICAL_CONSTANT_PI * delay * freq);
                const CF64 phasor = std::exp(phasorsExp);

                for (U64 p = 0; p < this->config.numberOfPolarizations; p++) {
                    const U64 polarizationOffset = p;

                    const U64 coefficientIndex = antennaCoeffOffset + frequencyCoeffOffset + polarizationOffset;
                    const U64 phasorsIndex = beamOffset + antennaPhasorOffset + frequencyPhasorOffset + polarizationOffset;

                    const auto coefficient = this->antennaCoefficients[coefficientIndex];
                    this->output.phasors[phasorsIndex] = phasor * coefficient;
                }
            }
        }
    }

    this->frequencyStepIndex = (this->frequencyStepIndex + 1) % this->frequencySteps;

    return Result::SUCCESS;
}

template class BLADE_API VLA<CF32>;
template class BLADE_API VLA<CF64>;

}  // namespace Blade::Modules::Phasor
