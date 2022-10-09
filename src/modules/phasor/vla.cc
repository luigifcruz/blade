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
    const auto coefficientUpchannelizedDims =
        config.antennaCoefficients.dims() * ArrayTensorDimensions({
            .A = 1,
            .F = config.preBeamformerChannelizerRate,
            .T = 1,
            .P = 1,
        });
    const auto coefficientExpectationRatio = coefficientUpchannelizedDims / this->getConfigCoefficientDims();
    if (coefficientExpectationRatio.size() == 0 || coefficientExpectationRatio.size() != coefficientExpectationRatio.numberOfFrequencyChannels()) {
        BL_FATAL("Number of antenna coefficients ({}) is not the expected size ({}), nor an integer multiple on the frequency axis.",
                coefficientUpchannelizedDims, this->getConfigCoefficientDims());
        BL_CHECK_THROW(Result::ERROR);
    }

    // TODO: check dimensions of the various config vectors

    this->frequencySteps = coefficientExpectationRatio.numberOfFrequencyChannels();
    this->frequencyStepIndex = 0;
    this->delayTimeIndex = 0;

    // Allocate output buffers.
    BL_CHECK_THROW(this->output.phasors.resize(getOutputPhasorsDims()));
    BL_CHECK_THROW(this->output.delays.resize(getOutputDelaysDims()));

    // Print configuration values.
    BL_INFO("Coefficient Dimensions [A, F*{}, T, P]: {}", config.preBeamformerChannelizerRate, coefficientUpchannelizedDims);
    BL_INFO("Delays Dimensions [B, A]: {} -> {}", this->getOutputDelays().dims(), "N/A");
    BL_INFO("Phasors Dimensions [B, A, F, T, P]: {} -> {}", "N/A", this->getOutputPhasors().dims());
    BL_INFO("Frequency steps: {}", this->frequencySteps);
    BL_INFO("Delay Times:")
    time_t curtime = 0;
    char timestr[32] = {0};
    for (U64 i = 0; i < config.delayTimes.size(); i++) {
        curtime = config.delayTimes[i];
        ctime_r(&curtime, timestr);
        timestr[strlen(timestr)-1] = '\0'; // Chop off trailing newline
        BL_INFO("   {}: {} ({})", i, config.delayTimes[i], timestr);
    }
}

template<typename OT>
const Result VLA<OT>::preprocess(const cudaStream_t& stream) {
    if (this->delayTimeIndex < this->config.delayTimes.size()-1) {
        const auto delayTimeThreshold = (3*this->config.delayTimes[this->delayTimeIndex+1] - this->config.delayTimes[this->delayTimeIndex]) / 2.0;
        const auto blockUnixS = calc_unix_sec_from_julian_date(this->input.blockJulianDate[0]);
        if (blockUnixS >= delayTimeThreshold) {
            this->delayTimeIndex += 1;
        }
    }

    const U64 currentFrequencyStepOffset = this->frequencyStepIndex * this->config.numberOfFrequencyChannels;
    const auto beamAntennaDelayDims = this->config.beamAntennaDelays.dims();
    const U64 currentDelayTimeOffset = this->delayTimeIndex * beamAntennaDelayDims.size()/beamAntennaDelayDims.numberOfTimeSamples();
    const U64 delayBeamStride = beamAntennaDelayDims.size()/(beamAntennaDelayDims.numberOfTimeSamples()*beamAntennaDelayDims.numberOfBeams());

    for (U64 b = 0; b < this->config.numberOfBeams; b++) {
        const U64 beamOffset = (b *
                                this->config.numberOfAntennas *
                                this->config.numberOfFrequencyChannels *
                                this->config.numberOfPolarizations);

        for (U64 a = 0; a < this->config.numberOfAntennas; a++) {
            const U64 antennaOffset = (a *
                                       this->config.numberOfFrequencyChannels *
                                       this->config.numberOfPolarizations);

            const F64 delay = this->config.beamAntennaDelays[currentDelayTimeOffset + b*delayBeamStride + a];

            for (U64 f = 0; f < this->config.numberOfFrequencyChannels; f++) {
                const U64 frequencyPhasorOffset = (f * this->config.numberOfPolarizations);
                const U64 frequencyCoeffOffset = (f + currentFrequencyStepOffset) * this->config.numberOfPolarizations;

                const F64 freq = this->config.channelZeroFrequencyHz + this->config.frequencyStartIndex * this->config.channelBandwidthHz + (f + currentFrequencyStepOffset) * this->config.channelBandwidthHz / this->config.preBeamformerChannelizerRate;
                const CF64 phasorsExp(0, -2 * BL_PHYSICAL_CONSTANT_PI * delay * freq);
                const CF64 phasor = std::exp(phasorsExp);

                for (U64 p = 0; p < this->config.numberOfPolarizations; p++) {
                    const U64 polarizationOffset = p;

                    const U64 coefficientIndex = (antennaOffset + frequencyCoeffOffset)/this->config.preBeamformerChannelizerRate + polarizationOffset;
                    const U64 phasorsIndex = beamOffset + antennaOffset + frequencyPhasorOffset + polarizationOffset;

                    const auto coefficient = this->config.antennaCoefficients[coefficientIndex];
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
