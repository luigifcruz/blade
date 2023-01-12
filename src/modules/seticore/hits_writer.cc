#define BL_LOG_DOMAIN "M::SETICORE::HITS_WRITER"

#include "blade/modules/seticore/hits_writer.hh"

#include "hits_writer.jit.hh"

namespace Blade::Modules::Seticore {

template<typename IT>
HitsWriter<IT>::HitsWriter(const Config& config, const Input& input)
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
const Result HitsWriter<IT>::process(const cudaStream_t& stream) {
    const auto inputDims = getInputBuffer().dims();
    const auto frequencyChannelByteStride = getInputBuffer().size_bytes() / (inputDims.numberOfAspects()*inputDims.numberOfFrequencyChannels());

    BL_DEBUG("Search found {} hits", input.hits.size());
    for (DedopplerHit hit : input.hits) {
        BL_DEBUG("\t{}", hit.toString());

        // Open output file.
        auto filepath = fmt::format("{}.seticore.{:04}.raw", this->config.filepathPrefix, this->fileId % 10000);
        this->fileDescriptor = open(filepath.c_str(), O_WRONLY | O_CREAT | (this->config.directio ? O_DIRECT : 0), 0644);
        if (this->fileDescriptor < 1) {
            BL_FATAL("Could not open '{}': {}\n", filepath, this->fileDescriptor);
            BL_CHECK_THROW(Result::ERROR);
        }

        this->gr_header.metadata.datashape.n_aspectchan = hit.highIndex() - hit.lowIndex();
        guppiraw_header_put_metadata(&this->gr_header);

        this->headerPut("DATATYPE", "FLOAT");
        this->headerPut("CHAN_BW", config.channelBandwidthHz);
        this->headerPut("TBIN", config.channelTimespanS);

        const auto& bytesWritten = guppiraw_write_block(
            this->fileDescriptor,
            &this->gr_header,
            this->input.buffer.data() + hit.lowIndex()*frequencyChannelByteStride
        );

        if (bytesWritten <= 0) {
            return Result::ERROR;
        }
        fileId += 1;
    }

    return Result::SUCCESS;
}

template class BLADE_API HitsWriter<CF16>;
template class BLADE_API HitsWriter<CF32>;

}  // namespace Blade::Modules::Seticore
