#define BL_LOG_DOMAIN "M::DEDOPPLER"

#include "blade/modules/seticore/dedoppler.hh"

#include "dedoppler.jit.hh"

namespace Blade::Modules::Seticore {

Dedoppler::Dedoppler(const Config& config, const Input& input)
        : Module(config.blockSize, dedoppler_kernel),
          config(config),
          input(input),
          dedopplerer(
            input.buf.dims().numberOfTimeSamples(),
            input.buf.dims().numberOfFrequencyChannels(),
            1e-6 * (this->config.observationBandwidthHz / input.buf.dims().numberOfFrequencyChannels()),
            1.0/(this->config.observationBandwidthHz / input.buf.dims().numberOfFrequencyChannels()),
            config.mitigateDcSpike
          ) {}

const Result Dedoppler::process(const cudaStream_t& stream) {
    const auto inputDims = this->input.buf.dims();
    FilterbankBuffer filterbankBuffer = FilterbankBuffer(inputDims.numberOfTimeSamples(), inputDims.numberOfFrequencyChannels(), this->input.buf.data());

    for (U64 beam = 0; beam < inputDims.numberOfAspects(); beam++) {
        for (U64 channel = 0; channel < inputDims.numberOfFrequencyChannels(); channel++) {
            vector<DedopplerHit> local_hits;
            
            dedopplerer.search(
                filterbankBuffer,
                beam,
                channel,
                this->config.maximumDriftrate,
                this->config.minimumDriftrate,
                this->config.snrThreshold,
                &local_hits
            );
        }
    }
    
    return Result::SUCCESS;
}

} // namespace Blade::Modules::Seticore
