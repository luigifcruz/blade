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
            this->config.channelTimespanS,
            config.mitigateDcSpike
          ) {

    this->metadata.has_dc_spike = config.mitigateDcSpike;
    this->metadata.coarse_channel_size = config.coarseChannelRate;

    BL_INFO("Dimensions [A, F, T, P]: {} -> {}", this->input.buf.dims(), "N/A");
    BL_INFO("Coarse Channel Rate: {}", this->config.coarseChannelRate);
    BL_INFO("Channel Bandwidth: {} Hz", this->config.channelBandwidthHz);
    BL_INFO("Channel Timespan: {} s", this->config.channelTimespanS);
}

void Dedoppler::setFrequencyOfFirstInputChannel(F64 hz) {
    this->metadata.fch1 = 1e-6 * hz;
}

const Result Dedoppler::process(const cudaStream_t& stream) {
    this->output.hits.clear();
    const auto inputDims = this->input.buf.dims();

    for (U64 beam = 0; beam < inputDims.numberOfAspects(); beam++) {
        FilterbankBuffer filterbankBuffer = FilterbankBuffer(
            inputDims.numberOfTimeSamples(),
            inputDims.numberOfFrequencyChannels(),
            this->input.buf.data() + beam*this->input.buf.size_bytes() / inputDims.numberOfAspects()
        );
        dedopplerer.search(
            filterbankBuffer,
            this->metadata,
            this->config.lastBeamIsIncoherent && beam + 1 == inputDims.numberOfAspects() ? -1 : beam,
            this->input.coarseFrequencyChannelOffset[0],
            this->config.maximumDriftRate,
            this->config.minimumDriftRate,
            this->config.snrThreshold,
            &this->output.hits
        );
    }

    return Result::SUCCESS;
}

} // namespace Blade::Modules::Seticore
