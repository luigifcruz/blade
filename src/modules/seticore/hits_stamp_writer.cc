#define BL_LOG_DOMAIN "M::SETICORE::HITS_STAMP_WRITER"

#include "blade/modules/seticore/hits_stamp_writer.hh"

#include "hits_writer.jit.hh"

namespace Blade::Modules::Seticore {

template<typename IT>
HitsStampWriter<IT>::HitsStampWriter(const Config& config, const Input& input)
        : Module(hits_writer_program),
          config(config),
          input(input),
          fileId(0),
          fileDescriptor(0) {

    FilterbankMetadata metadata;
    metadata.source_name = this->config.sourceName;
    metadata.fch1 = this->input.frequencyOfFirstInputChannelHz[0]*1e-6; // MHz
    metadata.foff = this->config.channelBandwidthHz*1e-6; // MHz
    metadata.tsamp = this->config.channelTimespanS;
    metadata.tstart = this->config.julianDateStart - 2400000.5; // from JD to MJD
    metadata.src_raj = this->config.phaseCenter.RA * 12.0 / BL_PHYSICAL_CONSTANT_PI; // hours
    metadata.src_dej = this->config.phaseCenter.DEC * 180.0 / BL_PHYSICAL_CONSTANT_PI; // degrees
    metadata.num_timesteps = this->config.totalNumberOfTimeSamples;
    metadata.num_channels = this->config.totalNumberOfFrequencyChannels;
    metadata.telescope_id = this->config.telescopeId;
    metadata.coarse_channel_size = this->config.coarseChannelRatio;
    metadata.num_coarse_channels = metadata.num_channels / metadata.coarse_channel_size;
    metadata.source_names = this->config.aspectNames;
    metadata.ras = std::vector<F64>();
    metadata.decs = std::vector<F64>();

    for (const RA_DEC& coord : this->config.aspectCoordinates) {
        metadata.ras.push_back(coord.RA * 12.0 / BL_PHYSICAL_CONSTANT_PI); // hours
        metadata.decs.push_back(coord.DEC * 180.0 / BL_PHYSICAL_CONSTANT_PI); // degrees
    }

    string output_filename = fmt::format("{}.seticore.hits", config.filepathPrefix);
    auto hfw = new HitFileWriter(output_filename, metadata);
    hfw->verbose = false;
    hit_recorder.reset(hfw);

    // Print configuration information.
    BL_INFO("Type: {} -> {}", TypeInfo<IT>::name, "N/A");
    BL_INFO("Dimensions [A, F, T, P]: {} -> {}", getInputBuffer().dims(), "N/A");
    BL_INFO("Output File Path: {}", config.filepathPrefix);
}

template<typename IT>
const Result HitsStampWriter<IT>::process(const cudaStream_t& stream) {
    const auto inputDims = getInputBuffer().dims();
    const auto frequencyChannelByteStride = getInputBuffer().size_bytes() / (inputDims.numberOfAspects()*inputDims.numberOfFrequencyChannels());

    vector<DedopplerHitGroup> groups = makeHitGroups(input.hits, this->config.hitsGroupingMargin);
    BL_DEBUG("{} group(s) of the search's {} hit(s)", groups.size(), input.hits.size());
    for (const DedopplerHitGroup& group : groups) {
        const DedopplerHit& top_hit = group.topHit();

        if (top_hit.drift_steps == 0) {
            // This is a vertical line. No drift = terrestrial. Skip it
            continue;
        }

        // Extract the stamp
        const int lowIndex = top_hit.lowIndex() - this->config.hitsGroupingMargin;
        const U64 first_channel = lowIndex < 0 ? 0 : (U64) lowIndex;
        const U64 highIndex = top_hit.highIndex() + (int)this->config.hitsGroupingMargin;
        const U64 last_channel = highIndex >= inputDims.numberOfFrequencyChannels() ? inputDims.numberOfFrequencyChannels()-1 : highIndex;
        
        BL_DEBUG("Top hit: {}", top_hit.toString());
        BL_DEBUG(
            "Extracting fine channels [{}, {}) from coarse channel {}",
            first_channel,
            last_channel,
            top_hit.coarse_channel
        );
        if (first_channel > last_channel) {
            BL_FATAL("First channel is larger than last: {} > {}", first_channel, last_channel);
            return Result::ASSERTION_ERROR;
        }
        const auto regionOfInterest = ArrayTensor<Device::CPU, IT>(
            getInputBuffer().data() + first_channel*frequencyChannelByteStride,
            {
                .A = inputDims.numberOfAspects(),
                .F = (U64) (last_channel - first_channel),
                .T = inputDims.numberOfTimeSamples(),
                .P = inputDims.numberOfPolarizations(),
            }
        );
        const auto regionOfInterestDims = regionOfInterest.dims();
        
        ::capnp::MallocMessageBuilder message;
        Stamp::Builder stamp = message.initRoot<Stamp>();
        stamp.setSeticoreVersion("0.0.0.a");
        stamp.setSourceName(this->config.sourceName);
        stamp.setRa(this->config.phaseCenter.RA * 12.0 / BL_PHYSICAL_CONSTANT_PI); // hours
        stamp.setDec(this->config.phaseCenter.DEC * 180.0 / BL_PHYSICAL_CONSTANT_PI); // degrees
        stamp.setFch1(this->input.frequencyOfFirstInputChannelHz[0]*1e-6); // MHz
        stamp.setFoff(this->config.channelBandwidthHz*1e-6); // MHz
        stamp.setTstart(this->config.julianDateStart); // TODO verify units
        stamp.setTsamp(this->config.channelTimespanS);
        stamp.setTelescopeId(this->config.telescopeId);
        stamp.setNumTimesteps(regionOfInterestDims.numberOfTimeSamples());
        stamp.setNumChannels(regionOfInterestDims.numberOfFrequencyChannels());
        stamp.setNumPolarizations(regionOfInterestDims.numberOfPolarizations());
        stamp.setNumAntennas(regionOfInterestDims.numberOfAspects());
        stamp.initData(2 * regionOfInterest.size());
        auto data = stamp.getData();

        for (int i = 0; i < (int) regionOfInterest.size(); ++i) {
            const auto value = regionOfInterest.data()[i];
            data.set(2 * i, value.real());
            data.set(2 * i + 1, value.imag());
        }

        stamp.setCoarseChannel(top_hit.coarse_channel);
        stamp.setFftSize(this->config.coarseChannelRatio);
        stamp.setStartChannel(top_hit.coarse_channel*this->config.coarseChannelRatio + first_channel);

        buildSignal(top_hit, stamp.getSignal());

        stamp.setSchan(this->config.coarseStartChannelIndex);
        stamp.setObsid(this->config.observationIdentifier);
        

        auto filepath = fmt::format("{}.seticore.{:04}.stamp", this->config.filepathPrefix, this->fileId % 10000);
        this->fileDescriptor = open(filepath.c_str(), O_WRONLY | O_CREAT, 0644);
        writeMessageToFd(this->fileDescriptor, message);
    
        close(this->fileDescriptor);
    }

    return Result::SUCCESS;
}

template class BLADE_API HitsStampWriter<CF32>;

}  // namespace Blade::Modules::Seticore
