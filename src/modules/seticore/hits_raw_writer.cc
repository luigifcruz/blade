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
        this->headerPut("FCH1", input.frequencyOfFirstChannelHz[0]);
        this->headerPut("CHAN_BW", config.channelBandwidthHz);
        this->headerPut("TBIN", config.channelTimespanS);
        const auto mjd = this->input.julianDateStart[0] - 2400000.5; // from JD to MJD
        const auto imjd = (U64) mjd;
        const auto smjd = (U64)((mjd - imjd) * 86400.0);
        const auto smjd_frac = ((mjd - imjd) * 86400.0) - smjd;
        this->headerPut("STT_IMJD", imjd);
        this->headerPut("STT_SMJD", smjd);
        this->headerPut("STT_OFFS", smjd_frac);

        const auto& bytesWritten = guppiraw_write_block(
            this->fileDescriptor,
            &this->gr_header,
            this->input.buffer.data() + first_channel*frequencyChannelByteStride
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
