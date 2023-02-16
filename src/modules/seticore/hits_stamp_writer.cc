#define BL_LOG_DOMAIN "M::SETICORE::HITS_STAMP_WRITER"

#include "blade/modules/seticore/hits_stamp_writer.hh"

#include "hits_writer.jit.hh"

namespace Blade::Modules::Seticore {

template<typename IT>
HitsStampWriter<IT>::HitsStampWriter(const Config& config, const Input& input)
        : Module(hits_writer_program),
          config(config),
          input(input),
          stampsWritten(0),
          fileId(0),
          fileDescriptor(0) {

    // Print configuration information.
    BL_INFO("Type: {} -> {}", TypeInfo<IT>::name, "N/A");
    BL_INFO("Dimensions [A, F, T, P]: {} -> {}", getInputBuffer().dims(), "N/A");
    BL_INFO("Output File Path: {}", config.filepathPrefix);
    BL_INFO("Excluding Zero Drift Rate Hits: {}", config.excludeDriftRateZero ? "YES" : "NO");
}

template<typename IT>
HitsStampWriter<IT>::~HitsStampWriter() {
    if (this->stampsWritten > 0) {
        close(this->fileDescriptor);
    }
    BL_DEBUG("Wrote {} stamp(s).", this->stampsWritten);
}

template<typename IT>
const Result HitsStampWriter<IT>::process(const cudaStream_t& stream) {
    if (input.hits.size() == 0) {
        return Result::SUCCESS;
    }

    const auto inputDims = getInputBuffer().dims();

    const int hitStampFrequencyMargin = this->config.channelBandwidthHz < 500.0 ? 500.0 / this->config.channelBandwidthHz : 1;

    // vector<DedopplerHitGroup> groups = makeHitGroups(input.hits, this->config.hitsGroupingMargin);
    // BL_DEBUG("{} group(s) of the search's {} hit(s)", groups.size(), input.hits.size());
    // for (const DedopplerHitGroup& group : groups) {
    //     const DedopplerHit& top_hit = group.topHit();
    for (const DedopplerHit& top_hit : input.hits) {

        if (this->config.excludeDriftRateZero && top_hit.drift_steps == 0) {
            continue;
        }

        // Extract the stamp
        const int lowIndex = top_hit.lowIndex() - hitStampFrequencyMargin;
        const U64 first_channel = lowIndex < 0 ? 0 : (U64) lowIndex;
        const U64 highIndex = top_hit.highIndex() + hitStampFrequencyMargin;
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
        const ArrayDimensions regionOfInterestDims = {
            .A = inputDims.numberOfAspects(),
            .F = (U64) (last_channel - first_channel),
            .T = inputDims.numberOfTimeSamples(),
            .P = inputDims.numberOfPolarizations(),
        };
        
        ::capnp::MallocMessageBuilder message;
        Stamp::Builder stamp = message.initRoot<Stamp>();
        stamp.setSeticoreVersion("0.0.0.a");
        stamp.setSourceName(this->config.sourceName);
        stamp.setRa(this->config.phaseCenter.RA * 12.0 / BL_PHYSICAL_CONSTANT_PI); // hours
        stamp.setDec(this->config.phaseCenter.DEC * 180.0 / BL_PHYSICAL_CONSTANT_PI); // degrees
        stamp.setFch1(
            (
                this->input.frequencyOfFirstChannelHz[0]
                + first_channel*this->config.channelBandwidthHz
            )
            * 1e-6
        ); // MHz
        stamp.setFoff(this->config.channelBandwidthHz*1e-6); // MHz
        stamp.setTstart(calc_unix_sec_from_julian_date(this->input.julianDateStart[0])); // JD -> Unix
        stamp.setTsamp(this->config.channelTimespanS);
        stamp.setTelescopeId(this->config.telescopeId);
        stamp.setNumTimesteps(regionOfInterestDims.numberOfTimeSamples());
        stamp.setNumChannels(regionOfInterestDims.numberOfFrequencyChannels());
        stamp.setNumPolarizations(regionOfInterestDims.numberOfPolarizations());
        stamp.setNumAntennas(regionOfInterestDims.numberOfAspects());
        stamp.initData(2 * regionOfInterestDims.size());
        auto data = stamp.getData();

        // AFTP -> TFPA
        for (int a = 0; a < (int) regionOfInterestDims.numberOfAspects(); a++) {
            for (int f = 0; f < (int) regionOfInterestDims.numberOfFrequencyChannels(); f++) {
                for (int t = 0; t < (int) regionOfInterestDims.numberOfTimeSamples(); t++) {
                    for (int p = 0; p < (int) regionOfInterestDims.numberOfPolarizations(); p++) {
                        const auto tfpa_index = (
                            ((
                            t*regionOfInterestDims.numberOfFrequencyChannels() + f
                            )*regionOfInterestDims.numberOfPolarizations() + p
                            )*regionOfInterestDims.numberOfAspects() + a
                        );
                        const auto value = getInputBuffer().data()[
                            ((
                            a*inputDims.numberOfFrequencyChannels() + f + first_channel
                            )*inputDims.numberOfTimeSamples() + t
                            )*inputDims.numberOfPolarizations() + p
                        ];
                        data.set(tfpa_index*2 + 0, value.real());
                        data.set(tfpa_index*2 + 1, value.imag());
                    }
                }
            }
        }

        stamp.setCoarseChannel(top_hit.coarse_channel);
        stamp.setFftSize(this->config.coarseChannelRatio);
        stamp.setStartChannel(first_channel);

        buildSignal(top_hit, stamp.getSignal());

        stamp.setSchan(this->config.coarseStartChannelIndex);
        stamp.setObsid(this->config.observationIdentifier);
        
        if (this->stampsWritten == 0) {
            auto filepath = fmt::format("{}.seticore.{:04}.stamps", this->config.filepathPrefix, this->fileId);
            this->fileDescriptor = open(filepath.c_str(), O_WRONLY | O_CREAT, 0644);
        }
        else if (this->stampsWritten % 500 == 0) {
            close(this->fileDescriptor);

            this->fileId += 1;
            auto filepath = fmt::format("{}.seticore.{:04}.stamps", this->config.filepathPrefix, this->fileId);
            this->fileDescriptor = open(filepath.c_str(), O_WRONLY | O_CREAT, 0644);
        }

        writeMessageToFd(this->fileDescriptor, message);
        this->stampsWritten += 1;
    }

    return Result::SUCCESS;
}

template class BLADE_API HitsStampWriter<CF32>;

}  // namespace Blade::Modules::Seticore
