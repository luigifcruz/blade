#define BL_LOG_DOMAIN "M::FILTERBANK::WRITER"

#include "blade/modules/filterbank/writer.hh"

#include "filterbank.jit.hh"

namespace Blade::Modules::Filterbank {

template<typename IT>
Writer<IT>::Writer(const Config& config, const Input& input) 
        : Module(config.blockSize, filterbank_kernel),
          config(config),
          input(input) {

    const auto inputDims = this->input.buffer.dims();
    this->fileDescriptors.resize(inputDims.numberOfAspects());

    // TODO negate foff and obs_bw
    this->filterbank_header = {
        .machine_id = this->config.machineId,
        .telescope_id = this->config.telescopeId,
        .data_type = 1,
        .barycentric = this->config.baryCentric,
        .pulsarcentric = this->config.pulsarCentric,
        .src_raj = this->config.sourceCoordinate.RA,
        .src_dej = this->config.sourceCoordinate.DEC,
        .az_start = this->config.azimuthStart,
        .za_start = this->config.zenithStart,
        .fch1 = this->config.firstChannelCenterFrequency,
        .foff = this->config.channelBandwidthHz,
        .nchans = (I32) inputDims.numberOfFrequencyChannels(),
        .nbeams = (I32) inputDims.numberOfAspects(),
        .ibeam = -1,
        .nbits = (I32) (this->input.buffer.size_bytes()/this->input.buffer.size())*8,
        .tstart = this->config.julianDateStart,
        .tsamp = 1.0/this->config.channelBandwidthHz,
        .nifs = this->config.numberOfIfChannels,
    };
    strncpy(this->filterbank_header.source_name, this->config.source_name.c_str(), 80);
    strncpy(this->filterbank_header.rawdatafile, this->config.rawdatafile.c_str(), 80);

    this->openFilesWriteHeaders();

    // Print configuration buffers.
    BL_INFO("Output File Path: {}", config.filepath);
    BL_INFO("Data Dimensions [B, F, T, P]: {} -> {}", inputDims, "N/A");
}

template<typename IT>
void Writer<IT>::openFilesWriteHeaders() {
    // Check configuration values.
    for (size_t i = 0; i < this->fileDescriptors.size(); i++) {
        auto filepath = fmt::format("{}-beam{:04}.fil", this->config.filepath, i % 10000);
        this->fileDescriptors[i] = open(filepath.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        this->filterbank_header.ibeam = i;
        filterbank_fd_write_padded_header(this->fileDescriptors[i], &this->filterbank_header, 1024);
    }
}

template<typename IT>
const Result Writer<IT>::preprocess(const cudaStream_t& stream) {
    // TODO shuffle from [batch, B, F, T, P] to [B, T, P, F]
    // TODO reverse frequencies

    const auto byteSize = this->input.buffer.size_bytes()/this->fileDescriptors.size();
    U64 bytesWritten = 0;

    for (size_t i = 0; i < this->fileDescriptors.size(); i++) {
        bytesWritten += write(
            this->fileDescriptors[i],
            this->input.buffer.data() + i*byteSize,
            byteSize
        );
    }

    return bytesWritten == this->input.buffer.size_bytes() ? Result::SUCCESS : Result::ERROR;
}

template class BLADE_API Writer<F16>;
template class BLADE_API Writer<F32>;
template class BLADE_API Writer<F64>;

}  // namespace Blade::Modules::Filterbank
