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
    this->metadata.fch1 = this->config.frequencyOfFirstChannelHz*1e-6; // MHz
    this->metadata.foff = this->config.channelBandwidthHz*1e-6; // MHz
    this->metadata.tsamp = this->config.channelTimespanS;
    this->metadata.tstart = 0.0; // from MJD, dynamically set
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

    BL_INFO("num_timesteps: {}",  this->metadata.num_timesteps);
    BL_INFO("num_channels: {}",  this->metadata.num_channels);
    BL_INFO("coarse_channel_size: {}",  this->metadata.coarse_channel_size);
    BL_INFO("num_coarse_channels: {}",  this->metadata.num_coarse_channels);

    BL_INFO("Dimensions [A, F, T, P]: {} -> {}", this->input.buf.dims(), "N/A");
    BL_INFO("Coarse Channel Rate: {}", this->config.coarseChannelRate);
    BL_INFO("Channel Bandwidth: {} Hz", this->config.channelBandwidthHz);
    BL_INFO("Channel Timespan: {} s", this->config.channelTimespanS);

    const auto t = input.buf.dims().numberOfTimeSamples();
    const auto t_previousPowerOf2 = t == 0 ? 0 : (0x80000000 >> __builtin_clz(t));
    if (t != t_previousPowerOf2) {
        BL_FATAL("Dedoppler must be provided a power of 2 timesamples!");
        BL_CHECK_THROW(Result::ASSERTION_ERROR);
    }
}

const Result Dedoppler::process(const cudaStream_t& stream) {
    this->output.hits.clear();
    const auto inputDims = this->input.buf.dims();
    const auto beamByteStride = this->input.buf.size() / inputDims.numberOfAspects();

    BL_CHECK(Memory::Copy(this->buf, this->input.buf, stream));
    this->metadata.tstart = this->input.julianDate[0] - 2400000.5; // from JD to MJD

    const auto skipLastBeam = this->config.lastBeamIsIncoherent & (!this->config.searchIncoherentBeam);
    const auto beamsToSearch = inputDims.numberOfAspects() - (skipLastBeam ? 1 : 0);
    BL_DEBUG("processing {} beams", beamsToSearch);
    for (U64 beam = 0; beam < beamsToSearch; beam++) {
        FilterbankBuffer beamFilterbankBuffer = FilterbankBuffer(
            inputDims.numberOfTimeSamples(),
            inputDims.numberOfFrequencyChannels(),
            this->input.buf.data() + beam*beamByteStride
        );
        dedopplerer.search(
            beamFilterbankBuffer,
            this->metadata,
            beam,
            this->input.coarseFrequencyChannelOffset[0],
            this->config.maximumDriftRate,
            this->config.minimumDriftRate,
            this->config.snrThreshold,
            &this->output.hits
        );
    }

    BL_CUDA_CHECK(cudaStreamSynchronize(stream), [&]{
        BL_FATAL("Failed to synchronize stream: {}", err);
    });

    if (this->config.lastBeamIsIncoherent) {
        FilterbankBuffer incohFilterbankBuffer = FilterbankBuffer(
            inputDims.numberOfTimeSamples(),
            inputDims.numberOfFrequencyChannels(),
            this->input.buf.data() + (inputDims.numberOfAspects()-1)*beamByteStride
        );
        dedopplerer.addIncoherentPower(incohFilterbankBuffer, this->output.hits);
    }

    for (const DedopplerHit& hit : this->output.hits) {
        BL_DEBUG("Hit: {}", hit.toString());
        hit_recorder->recordHit(hit, this->buf.data());
    }

    return Result::SUCCESS;
}

} // namespace Blade::Modules::Seticore
