#define BL_LOG_DOMAIN "M::DEDOPPLER"

#include "blade/modules/seticore/dedoppler.hh"

#include "dedoppler.jit.hh"

namespace Blade::Modules::Seticore {

Dedoppler::Dedoppler(const Config& config, const Input& input)
        : Module(dedoppler_program),
          config(config),
          input(input),
          dedopplerer(
            input.buf.dims().numberOfTimeSamples(),
            input.buf.dims().numberOfFrequencyChannels(),
            1e-6 * this->config.channelBandwidthHz,
            1.0/this->config.channelBandwidthHz,
            config.mitigateDcSpike
          ) {
    BL_INFO("Channel Bandwidth: {} Hz", this->config.channelBandwidthHz);
    BL_INFO("Dimensions [A, F, T, P]: {} -> {}", this->input.buf.dims(), "N/A");
}

const Result Dedoppler::process(const cudaStream_t& stream) {
    this->output.hits.clear();
    const auto inputDims = this->input.buf.dims();
    FilterbankBuffer filterbankBuffer = FilterbankBuffer(inputDims.numberOfTimeSamples(), inputDims.numberOfFrequencyChannels(), this->input.buf.data());

    for (U64 beam = 0; beam < inputDims.numberOfAspects(); beam++) {
        for (U64 channel = 0; channel < inputDims.numberOfFrequencyChannels(); channel++) {

            dedopplerer.search(
                filterbankBuffer,
                beam,
                channel,
                this->config.maximumDriftRate,
                this->config.minimumDriftRate,
                this->config.snrThreshold,
                &this->output.hits
            );

        }
    }

    return Result::SUCCESS;
}

} // namespace Blade::Modules::Seticore
