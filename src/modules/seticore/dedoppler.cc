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

    this->metadata.source_name = this->config.sourceName;
    this->metadata.fch1 = 0.0; // MHz
    this->metadata.foff = this->config.channelBandwidthHz*1e-6; // MHz
    this->metadata.tsamp = this->config.channelTimespanS;
    this->metadata.tstart = this->config.julianDateStart - 2400000.5; // from JD to MJD
    this->metadata.src_raj = this->config.phaseCenter.RA * 12.0 / BL_PHYSICAL_CONSTANT_PI; // hours
    this->metadata.src_dej = this->config.phaseCenter.DEC * 180.0 / BL_PHYSICAL_CONSTANT_PI; // degrees
    this->metadata.num_timesteps = input.buf.dims().numberOfTimeSamples();
    this->metadata.num_channels = input.buf.dims().numberOfFrequencyChannels();
    this->metadata.telescope_id = this->config.telescopeId;
    this->metadata.coarse_channel_size = this->config.coarseChannelRate;
    this->metadata.num_coarse_channels = metadata.num_channels / metadata.coarse_channel_size;
    this->metadata.source_names = this->config.aspectNames;
    this->metadata.ras = std::vector<F64>();
    this->metadata.decs = std::vector<F64>();
    this->metadata.has_dc_spike = config.mitigateDcSpike;

    for (const RA_DEC& coord : this->config.aspectCoordinates) {
        this->metadata.ras.push_back(coord.RA * 12.0 / BL_PHYSICAL_CONSTANT_PI); // hours
        this->metadata.decs.push_back(coord.DEC * 180.0 / BL_PHYSICAL_CONSTANT_PI); // degrees
    }

    string output_filename = fmt::format("{}.seticore.hits", config.filepathPrefix);
    auto hfw = new HitFileWriter(output_filename, metadata);
    hfw->verbose = false;
    hit_recorder.reset(hfw);

    this->buf.resize(this->input.buf.dims());

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
    const auto beamByteStride = this->input.buf.size_bytes() / inputDims.numberOfAspects();

    BL_CHECK(Memory::Copy(this->buf, this->input.buf, stream));

    const auto beamsToSearch = inputDims.numberOfAspects() - (this->config.lastBeamIsIncoherent ? 1 : 0);
    BL_DEBUG("processing {} beams", beamsToSearch);
    for (U64 beam = 0; beam < beamsToSearch; beam++) {
        FilterbankBuffer filterbankBuffer = FilterbankBuffer(
            inputDims.numberOfTimeSamples(),
            inputDims.numberOfFrequencyChannels(),
            this->input.buf.data() + beam*beamByteStride
        );
        dedopplerer.search(
            filterbankBuffer,
            this->metadata,
            false,
            this->input.coarseFrequencyChannelOffset[0],
            this->config.maximumDriftRate,
            this->config.minimumDriftRate,
            this->config.snrThreshold,
            &this->output.hits
        );
    }

    if (this->config.lastBeamIsIncoherent) {
        FilterbankBuffer filterbankBuffer = FilterbankBuffer(
            inputDims.numberOfTimeSamples(),
            inputDims.numberOfFrequencyChannels(),
            this->input.buf.data() + (inputDims.numberOfAspects()-1)*beamByteStride
        );
        
        dedopplerer.addIncoherentPower(filterbankBuffer, this->output.hits);
    }

    BL_CUDA_CHECK(cudaStreamSynchronize(stream), [&]{
        BL_FATAL("Failed to synchronize stream: {}", err);
    });
    
    for (const DedopplerHit& hit : this->output.hits) {
        hit_recorder->recordHit(hit, this->buf.data());
    }

    return Result::SUCCESS;
}

} // namespace Blade::Modules::Seticore
