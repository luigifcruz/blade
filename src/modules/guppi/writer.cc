#define BL_LOG_DOMAIN "M::GUPPI::WRITER"

#include "blade/modules/guppi/writer.hh"

#include "guppi.jit.hh"

namespace Blade::Modules::Guppi {

template<typename IT>
Writer<IT>::Writer(const Config& config, 
                   const Input& input,
                   const cudaStream_t& stream)
        : Module(guppi_program),
          config(config),
          input(input),
          fileId(0),
          writeCounter(0),
          fileDescriptor(0) {
    // Open output file.
    auto filepath = fmt::format("{}.{:04}.raw", this->config.filepath, this->fileId % 10000);
    this->fileDescriptor = open(filepath.c_str(), O_WRONLY | O_CREAT | (this->config.directio ? O_DIRECT : 0), 0644);
    if (this->fileDescriptor < 1) {
        BL_FATAL("Could not open '{}': {}\n", filepath, this->fileDescriptor);
        BL_CHECK_THROW(Result::ERROR);
    }

    // Add expected metadata to the header.
    this->gr_header.metadata.datashape.n_aspect = getInputBuffer().shape().numberOfAspects();
    this->gr_header.metadata.datashape.n_aspectchan = getInputBuffer().shape().numberOfFrequencyChannels();
    this->gr_header.metadata.datashape.n_time = getInputBuffer().shape().numberOfTimeSamples();
    this->gr_header.metadata.datashape.n_pol = getInputBuffer().shape().numberOfPolarizations();
    this->gr_header.metadata.datashape.n_bit = sizeof(IT) * 8 / 2;
    this->gr_header.metadata.directio = this->config.directio ? 1 : 0;
    guppiraw_header_put_metadata(&this->gr_header);

    // Add custom metadata to the header.
    this->headerPut("NBEAM", getInputBuffer().shape().numberOfAspects());
    this->headerPut("DATATYPE", "FLOAT");

    // Print configuration information.
    BL_INFO("Type: {} -> {}", TypeInfo<IT>::name, "N/A");
    BL_INFO("Shape: {} -> {}", getInputBuffer().shape(), "N/A");
    BL_INFO("Output File Path: {}", config.filepath);
    BL_INFO("Direct I/O: {}", config.directio ? "YES" : "NO");
}

template<typename IT>
Result Writer<IT>::preprocess(const cudaStream_t& stream,
                              const U64& currentComputeCount) {
    const auto& bytesWritten = guppiraw_write_block_batched(
                                    this->fileDescriptor, 
                                    &this->gr_header, 
                                    this->input.buffer.data(),
                                    1,
                                    this->config.inputFrequencyBatches);

    if (bytesWritten <= 0) {
        return Result::ERROR;
    }

    this->headerPut("PKTIDX", getInputBuffer().shape().numberOfTimeSamples() * writeCounter++);

    return Result::SUCCESS;
}

template class BLADE_API Writer<CF16>;
template class BLADE_API Writer<CF32>;

}  // namespace Blade::Modules::Guppi
