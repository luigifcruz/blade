#define BL_LOG_DOMAIN "M::SETICORE::HITS_RAW_WRITER"

#include "blade/modules/seticore/hits_raw_writer.hh"

#include "hits_writer.jit.hh"

namespace Blade::Modules::Seticore {

template<typename IT>
HitsRawWriter<IT>::HitsRawWriter(const Config& config, const Input& input)
        : Module(hits_writer_program),
          config(config),
          input(input),
          fileId(0),
          fileDescriptor(0) {

    // Set fundamental datashape elements
    this->gr_header.metadata.datashape.n_aspect = getInputBuffer().dims().numberOfAspects();
    this->gr_header.metadata.datashape.n_aspectchan = getInputBuffer().dims().numberOfFrequencyChannels();
    this->gr_header.metadata.datashape.n_time = getInputBuffer().dims().numberOfTimeSamples();
    this->gr_header.metadata.datashape.n_pol = getInputBuffer().dims().numberOfPolarizations();
    this->gr_header.metadata.datashape.n_bit = sizeof(IT) * 8 / 2;
    this->gr_header.metadata.directio = this->config.directio ? 1 : 0;

    // Print configuration information.
    BL_INFO("Type: {} -> {}", TypeInfo<IT>::name, "N/A");
    BL_INFO("Dimensions [A, F, T, P]: {} -> {}", getInputBuffer().dims(), "N/A");
    BL_INFO("Output File Path: {}", config.filepathPrefix);
    BL_INFO("Direct I/O: {}", config.directio ? "YES" : "NO");
}

template<typename IT>
const Result HitsRawWriter<IT>::process(const cudaStream_t& stream) {
    if (input.hits.size() == 0) {
        BL_DEBUG("No hits.");
        return Result::SUCCESS;
    }

    const auto inputDims = getInputBuffer().dims();
    const auto frequencyChannelStride = getInputBuffer().size() / (inputDims.numberOfAspects()*inputDims.numberOfFrequencyChannels());

    int hitStampFrequencyMargin = 1;
    if (this->config.stampFrequencyMarginHz <= 0.0) {
        hitStampFrequencyMargin = 0;
    }
    else if (abs(this->config.channelBandwidthHz) < this->config.stampFrequencyMarginHz) {
        hitStampFrequencyMargin = this->config.stampFrequencyMarginHz / abs(this->config.channelBandwidthHz);
    }

    vector<DedopplerHitGroup> groups = makeHitGroups(input.hits, this->config.hitsGroupingMargin);
    // The grouping mechanism ensures all hits are in a single group.
    // The groups bounds are transitive, expanding with each new-comer
    // Because of this, some groups could have overlapping channel-spans.
    // The real objective is just to ensure that all frequency data that
    // resulted in a hit is stamped out. So leave them be, but don't stamp
    // frequency ranges twice...
    
    vector<U64> stamps_first_channel;
    vector<U64> stamps_last_channel;
    stamps_first_channel.reserve(groups.size());
    stamps_last_channel.reserve(groups.size());
    BL_DEBUG("{} group(s) of the search's {} hit(s), stamps are padded by {} frequency-channels.", groups.size(), input.hits.size(), hitStampFrequencyMargin);
    for (const DedopplerHitGroup& group : groups) {
        // Consider the group
        //  a stamp covers the extremes of all the hits
        //  + the hitStampFrequencyMargin
        int lowIndex = inputDims.numberOfFrequencyChannels();
        int highIndex = 0;
        for (const DedopplerHit& hit : group.hits) {
            if (hit.lowIndex() < lowIndex) {
                lowIndex = hit.lowIndex();
            }
            if (hit.highIndex() > highIndex) {
                highIndex = hit.highIndex();
            }
        }

        lowIndex -= hitStampFrequencyMargin;
        highIndex += hitStampFrequencyMargin;
        const U64 first_channel = lowIndex < 0 ? 0 : (U64) lowIndex;
        const U64 last_channel = highIndex > inputDims.numberOfFrequencyChannels() ? inputDims.numberOfFrequencyChannels() : (U64) highIndex;

        BL_DEBUG(
            "Group channel range spans [{}, {}].",
            first_channel,
            last_channel
        );
        if (first_channel > last_channel) {
            BL_FATAL("Channels are wrong way around");
            BL_CHECK_THROW(Result::ERROR);
        }
        
        // double check that this channel range has not already been stamped
        bool stamp_is_novel = true;
        for (size_t index = 0; stamp_is_novel && index < stamps_first_channel.size(); index++) {
            stamp_is_novel &= !(stamps_first_channel.at(index) <= first_channel
                && stamps_last_channel.at(index) >= last_channel);
            
            if (!stamp_is_novel) {
                BL_DEBUG(
                    "Group channel range covered by previous stamp #{} spanning [{}, {}]",
                    index,
                    stamps_first_channel.at(index),
                    stamps_last_channel.at(index)
                );
            }
        }
        if (!stamp_is_novel) {
            continue;
        }
        
        // Extract the stamp
        const DedopplerHit& top_hit = group.topHit();
        BL_DEBUG("Stamp #{} of group with top hit: {}", stamps_first_channel.size(), top_hit.toString());
        stamps_first_channel.emplace_back(first_channel);
        stamps_last_channel.emplace_back(last_channel);

        // Open output file.
        auto filepath = fmt::format("{}.seticore.{:04}.raw", this->config.filepathPrefix, this->fileId % 10000);
        this->fileDescriptor = open(filepath.c_str(), O_WRONLY | O_CREAT | (this->config.directio ? O_DIRECT : 0), 0644);
        if (this->fileDescriptor < 1) {
            BL_FATAL("Could not open '{}': {}\n", filepath, this->fileDescriptor);
            BL_CHECK_THROW(Result::ERROR);
        }

        this->gr_header.metadata.datashape.n_aspectchan = last_channel - first_channel;
        guppiraw_header_put_metadata(&this->gr_header);

        this->headerPut("DATATYPE", "FLOAT");
        this->headerPut("SRC_NAME", config.sourceName);
        this->headerPut("TELESCID", config.telescopeId);
        this->headerPut("OBSID", config.observationIdentifier);
        this->headerPut("RA_STR", config.phaseCenter.RA * 12.0 / BL_PHYSICAL_CONSTANT_PI); // hours
        this->headerPut("DEC_STR", config.phaseCenter.DEC * 180.0 / BL_PHYSICAL_CONSTANT_PI); // degrees
        this->headerPut("SCHANORG", config.coarseStartChannelIndex);
        this->headerPut("FCH1", input.frequencyOfFirstChannelHz[0] + config.channelBandwidthHz*first_channel);
        this->headerPut("CHAN_BW", config.channelBandwidthHz);
        this->headerPut("TBIN", config.channelTimespanS);
        const auto mjd = this->input.julianDateStart[0] - 2400000.5; // from JD to MJD
        const auto imjd = (U64) mjd;
        const auto smjd = (U64)((mjd - imjd) * 86400.0);
        const auto smjd_frac = ((mjd - imjd) * 86400.0) - smjd;
        this->headerPut("STT_IMJD", imjd);
        this->headerPut("STT_SMJD", smjd);
        this->headerPut("STT_OFFS", smjd_frac);

        // TODO fix this, have to stride aspects due to the channel selection
        const auto& bytesWritten = guppiraw_write_block(
            this->fileDescriptor,
            &this->gr_header,
            this->input.buffer.data() + first_channel*frequencyChannelStride
        );

        close(this->fileDescriptor);
        if (bytesWritten <= 0) {
            return Result::ERROR;
        }
        this->fileId += 1;
    }

    return Result::SUCCESS;
}

template class BLADE_API HitsRawWriter<CF16>;
template class BLADE_API HitsRawWriter<CF32>;

}  // namespace Blade::Modules::Seticore
