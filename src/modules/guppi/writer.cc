#include "blade/modules/guppi/writer.hh"

#include "guppi.jit.hh"

namespace Blade::Modules::Guppi {

template<typename IT>
Writer<IT>::Writer(const Config& config)
        : Module(config.blockSize, guppi_kernel),
          config(config) {
    BL_INFO("===== GUPPI Writer Module Configuration");

    BL_CHECK_THROW(InitInput(this->input.buf, getTotalInputSize()));

    auto filepath = fmt::format("{}.{:04}.raw", this->config.filepath, this->file_id % 10000);
    this->file_descriptor = open(filepath.c_str(), O_WRONLY | O_CREAT | (this->config.directio ? O_DIRECT : 0), 0644);
    if (this->file_descriptor < 1) {
        BL_FATAL("Could not open '{}': {}\n", filepath, this->file_descriptor);
        BL_CHECK_THROW(Result::ERROR);
    }

    this->gr_header.metadata.datashape.n_beam = this->getNumberOfBeams();
    this->gr_header.metadata.datashape.n_ant = this->getNumberOfAntennas();
    this->gr_header.metadata.datashape.n_aspectchan = this->getTotalNumberOfFrequencyChannels();
    this->gr_header.metadata.datashape.n_time = this->getNumberOfTimeSamples();
    this->gr_header.metadata.datashape.n_pol = this->getNumberOfPolarizations();
    this->gr_header.metadata.datashape.n_bit = sizeof(IT) * 8 / 2;
    this->gr_header.metadata.directio = this->config.directio ? 1 : 0;
    guppiraw_header_put_metadata(&this->gr_header);
    this->headerPut("DATATYPE", "FLOAT");

    BL_INFO("Output File Path: {}", config.filepath);
    BL_INFO("Direct I/O: {}", config.directio ? "ENABLED" : "DISABLED")
    BL_INFO("Number of Batches: {}", this->getNumberOfBatches());
    BL_INFO("Step Number of Beams: {}", this->getNumberOfBeams());
    BL_INFO("Step Number of Antennas: {}", this->getNumberOfAntennas());
    BL_INFO("Step Number of Frequency Channels: {}", this->getNumberOfFrequencyChannels());
    BL_INFO("Step Number of Time Samples: {}", this->getNumberOfTimeSamples());
    BL_INFO("Step Number of Polarizations: {}", this->getNumberOfPolarizations());
    BL_INFO("Total Number of Frequency Channels: {}", this->getTotalNumberOfFrequencyChannels());
}

template<typename IT>
Result Writer<IT>::preprocess(const cudaStream_t& stream) {
    const auto& bytesWritten = 
        guppiraw_write_block_batched(
            this->file_descriptor, 
            &this->gr_header, 
            this->input.buf.data(), 
            1, 
            this->getNumberOfBatches());

    if (bytesWritten <= 0) {
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

template class BLADE_API Writer<CF16>;
template class BLADE_API Writer<CF32>;

}  // namespace Blade::Modules::Guppi
