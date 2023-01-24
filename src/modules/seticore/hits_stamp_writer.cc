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
        stamp.setTstart(calc_unix_sec_from_julian_date(this->config.julianDateStart)); // JD -> Unix
        stamp.setTsamp(this->config.channelTimespanS);
        stamp.setTelescopeId(this->config.telescopeId);
        stamp.setNumTimesteps(regionOfInterestDims.numberOfTimeSamples());
        stamp.setNumChannels(regionOfInterestDims.numberOfFrequencyChannels());
        stamp.setNumPolarizations(regionOfInterestDims.numberOfPolarizations());
        stamp.setNumAntennas(regionOfInterestDims.numberOfAspects());
        stamp.initData(2 * regionOfInterest.size());
        auto data = stamp.getData();

        // AFTP -> TFPA
        int aftp_index = 0;
        const auto stamp_timeStride = regionOfInterest.size()*2/regionOfInterestDims.numberOfTimeSamples();
        const auto stamp_frequencyStride = stamp_timeStride/regionOfInterestDims.numberOfFrequencyChannels();
        const auto stamp_polarizationStride = stamp_frequencyStride/regionOfInterestDims.numberOfPolarizations();
        const auto stamp_aspectStride = stamp_polarizationStride/regionOfInterestDims.numberOfAspects();

        for (int a = 0; a < (int) regionOfInterestDims.numberOfAspects(); a++) {
            for (int f = 0; f < (int) regionOfInterestDims.numberOfFrequencyChannels(); f++) {
                for (int t = 0; t < (int) regionOfInterestDims.numberOfTimeSamples(); t++) {
                    for (int p = 0; p < (int) regionOfInterestDims.numberOfPolarizations(); p++) {
                        const auto tfpa_index = (
                            t*stamp_timeStride
                            + f*stamp_frequencyStride
                            + p*stamp_polarizationStride
                            + a*stamp_aspectStride
                        );
                        const auto value = regionOfInterest.data()[aftp_index];
                        data.set(tfpa_index + 0, value.real());
                        data.set(tfpa_index + 1, value.imag());
                        aftp_index += 1;
                    }
                }
            }
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
