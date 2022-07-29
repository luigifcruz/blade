#include "blade/modules/guppi/writer.hh"

#include "guppi.jit.hh"

namespace Blade::Modules::Guppi {

template<typename IT>
Writer<IT>::Writer(const Config& config)
        : Module(config.blockSize, guppi_kernel),
          config(config) {
    BL_INFO("===== GUPPI Writer Module Configuration");

    BL_CHECK_THROW(InitInput(this->input.buf, getInputSize()));

    char* filepath = (char*) malloc(this->config.filepathStem.length()+10);
    sprintf(filepath, "%s.%04ld.raw", this->config.filepathStem.c_str(), this->file_id%10000);
    this->file_descriptor = open(filepath, O_WRONLY|O_CREAT| (this->config.directio ? O_DIRECT : 0), 0644);
    if (this->file_descriptor < 1) {
        BL_FATAL("Could not open '{}': {}\n", filepath, this->file_descriptor);
    }
    free(filepath);

    this->gr_header.metadata.datashape.n_beam = this->config.numberOfBeams;
    this->gr_header.metadata.datashape.n_ant = this->config.numberOfAntennas;
    this->gr_header.metadata.datashape.n_aspectchan = this->getTotalNumberOfFrequencyChannels();
    this->gr_header.metadata.datashape.n_time = this->getNumberOfTimeSamples();
    this->gr_header.metadata.datashape.n_pol = this->getNumberOfPolarizations();
    this->gr_header.metadata.datashape.n_bit = sizeof(IT)*8/2;
    this->gr_header.metadata.directio = this->config.directio ? 1 : 0;
    guppiraw_header_put_metadata(&this->gr_header);
    this->headerPut("DATATYPE", "FLOAT");
}

template class BLADE_API Writer<CF16>;
template class BLADE_API Writer<CF32>;

}  // namespace Blade::Modules::Guppi