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

    this->filterbank_header = {
        .machine_id = this->config.machineId,
        .telescope_id = filterbank_telescope_id(this->config.telescopeName.c_str()),
        .data_type = 1,
        .barycentric = this->config.baryCentric,
        .pulsarcentric = this->config.pulsarCentric,
        .src_raj = this->config.sourceCoordinate.RA * 24.0/360.0, // from degrees to hours
        .src_dej = this->config.sourceCoordinate.DEC,
        .az_start = this->config.azimuthStart,
        .za_start = this->config.zenithStart,
        .fch1 = 1e-6 * (this->config.observationFrequencyHz - this->config.observationBandwidthHz * (inputDims.numberOfFrequencyChannels()-1)/(2*inputDims.numberOfFrequencyChannels())),
        .foff = 1e-6 * (this->config.observationBandwidthHz / inputDims.numberOfFrequencyChannels()),
        .nchans = (I32) inputDims.numberOfFrequencyChannels(),
        .nbeams = (I32) inputDims.numberOfAspects(),
        .ibeam = -1,
        .nbits = (I32) sizeof(IT)*8,
        .tstart = this->config.julianDateStart - 2400000.5, // from JD to MJD
        .tsamp = abs(1.0/(this->config.observationBandwidthHz / inputDims.numberOfFrequencyChannels())), // time always moves forward
        .nifs = this->config.numberOfIfChannels,
    };
    strncpy(this->filterbank_header.source_name, this->config.sourceName.c_str(), 80);
    strncpy(this->filterbank_header.rawdatafile, this->config.sourceDataFilename.c_str(), 80);

    this->openFilesWriteHeaders();

    // Print configuration buffers.
    BL_INFO("Output File Path: {}", config.filepath);
    BL_INFO("Frequency Batches per Input: {}", config.numberOfInputFrequencyChannelBatches);
    BL_INFO("Data Dimensions [B, F, T, P]: {} -> {}", inputDims, "N/A");
}

template<typename IT>
void Writer<IT>::openFilesWriteHeaders() {
    // Check configuration values.
    for (size_t i = 0; i < this->fileDescriptors.size(); i++) {
        auto filepath = fmt::format("{}-beam{:04}.fil", this->config.filepath, i % 10000);
        this->fileDescriptors[i] = open(filepath.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (this->fileDescriptors[i] < 1) {
            BL_FATAL("Could not open '{}': {}\n", filepath, this->fileDescriptors[i]);
            BL_CHECK_THROW(Result::ERROR);
        }
        this->filterbank_header.ibeam = i;

        filterbank_fd_write_header(this->fileDescriptors[i], &this->filterbank_header);
    }
}

template<typename IT>
const Result Writer<IT>::preprocess(const cudaStream_t& stream) {
    // Expect data with dimensions: [Fbatch, B, T, P, F]

    const auto inputDims = this->input.buffer.dims();
    const U64 numberOfTimePolarizationSamples = inputDims.numberOfTimeSamples()*inputDims.numberOfPolarizations();
    const U64 frequencyBatchByteStride = this->input.buffer.size_bytes()/this->config.numberOfInputFrequencyChannelBatches;
    const U64 aspectByteStride = frequencyBatchByteStride/inputDims.numberOfAspects();
    const U64 timepolSampleByteStride = (inputDims.numberOfFrequencyChannels()/this->config.numberOfInputFrequencyChannelBatches)*sizeof(IT);

    U64 bytesWritten = 0;
    const long max_iovecs = sysconf(_SC_IOV_MAX);
    
    struct iovec* iovecs = (struct iovec*) malloc(max_iovecs * sizeof(struct iovec));
    int iovec_count = 0;

    for (size_t a = 0; a < inputDims.numberOfAspects(); a++) {
        if (this->config.numberOfInputFrequencyChannelBatches == 1) {
            bytesWritten += write(
                this->fileDescriptors[a],
                this->input.buffer.data() + a*aspectByteStride/sizeof(IT),
                aspectByteStride
            );
        } else {
            for (size_t tp = 0; tp < numberOfTimePolarizationSamples; tp++) {
                for (size_t fb = 0; fb < this->config.numberOfInputFrequencyChannelBatches; fb++) {
                    iovecs[iovec_count].iov_base = this->input.buffer.data() +
                        ( fb * frequencyBatchByteStride
                        + a * aspectByteStride
                        + tp * timepolSampleByteStride
                        )/sizeof(IT);
                    iovecs[iovec_count].iov_len = timepolSampleByteStride;
                    iovec_count++;
                    if(iovec_count == max_iovecs) {
                        bytesWritten += writev(this->fileDescriptors[a], iovecs, iovec_count);
                        iovec_count = 0;
                    }
                }
            }
            if (iovec_count > 0) {
                bytesWritten += writev(this->fileDescriptors[a], iovecs, iovec_count);
                iovec_count = 0;
            }
        }
    }
    free(iovecs);

    return bytesWritten == this->input.buffer.size_bytes() ? Result::SUCCESS : Result::ERROR;
}

template class BLADE_API Writer<F16>;
template class BLADE_API Writer<F32>;
template class BLADE_API Writer<F64>;

}  // namespace Blade::Modules::Filterbank
