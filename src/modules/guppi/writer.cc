#include "blade/modules/guppi/writer.hh"

#include "guppi.jit.hh"

namespace Blade::Modules::Guppi {

template<typename IT>
Writer<IT>::Writer(const Config& config, const Input& input)
        : Module(config.blockSize, guppi_kernel),
          config(config),
          input(input),
          fileId(0),
          writeCounter(0),
          fileDescriptor(0) {
    BL_INFO("===== GUPPI Writer Module Configuration");

    auto filepath = fmt::format("{}.{:04}.raw", this->config.filepath, this->fileId % 10000);
    this->fileDescriptor = open(filepath.c_str(), O_WRONLY | O_CREAT | (this->config.directio ? O_DIRECT : 0), 0644);
    if (this->fileDescriptor < 1) {
        BL_FATAL("Could not open '{}': {}\n", filepath, this->fileDescriptor);
        BL_CHECK_THROW(Result::ERROR);
    }

    this->gr_header.metadata.datashape.n_beam = this->getStepNumberOfBeams();
    this->gr_header.metadata.datashape.n_ant = this->getStepNumberOfAntennas();
    this->gr_header.metadata.datashape.n_aspectchan = this->getTotalNumberOfFrequencyChannels();
    this->gr_header.metadata.datashape.n_time = this->getStepNumberOfTimeSamples();
    this->gr_header.metadata.datashape.n_pol = this->getStepNumberOfPolarizations();
    this->gr_header.metadata.datashape.n_bit = sizeof(IT) * 8 / 2;
    this->gr_header.metadata.directio = this->config.directio ? 1 : 0;
    guppiraw_header_put_metadata(&this->gr_header);
    this->headerPut("DATATYPE", "FLOAT");

    BL_INFO("Output File Path: {}", config.filepath);
    BL_INFO("Direct I/O: {}", config.directio ? "YES" : "NO")
    BL_INFO("Number of Steps: {}", this->getNumberOfSteps());
    BL_INFO("Step Number of Beams: {}", this->getStepNumberOfBeams());
    BL_INFO("Step Number of Antennas: {}", this->getStepNumberOfAntennas());
    BL_INFO("Step Number of Frequency Channels: {}", this->getStepNumberOfFrequencyChannels());
    BL_INFO("Step Number of Time Samples: {}", this->getStepNumberOfTimeSamples());
    BL_INFO("Step Number of Polarizations: {}", this->getStepNumberOfPolarizations());
    BL_INFO("Total Number of Beams: {}", this->getTotalNumberOfBeams());
    BL_INFO("Total Number of Antennas: {}", this->getTotalNumberOfAntennas());
    BL_INFO("Total Number of Frequency Channels: {}", this->getTotalNumberOfFrequencyChannels());
    BL_INFO("Total Number of Time Samples: {}", this->getTotalNumberOfTimeSamples());
    BL_INFO("Total Number of Polarizations: {}", this->getTotalNumberOfPolarizations());

    BL_CHECK_THROW(InitInput(this->input.totalBuffer, getTotalInputBufferSize()));
}

template<typename IT>
Result Writer<IT>::preprocess(const cudaStream_t& stream) {
    const auto& bytesWritten = 
        guppiraw_write_block_batched(
            this->fileDescriptor, 
            &this->gr_header, 
            this->input.totalBuffer.data(), 
            1, 
            this->getNumberOfSteps());

    if (bytesWritten <= 0) {
        return Result::ERROR;
    }

    this->headerPut("PKTIDX", this->getStepNumberOfTimeSamples() * writeCounter++);

    return Result::SUCCESS;
}

template class BLADE_API Writer<CF16>;
template class BLADE_API Writer<CF32>;

}  // namespace Blade::Modules::Guppi
