#define BL_LOG_DOMAIN "M::FILTERBANK::WRITER"

#include "blade/modules/filterbank/writer.hh"

#include "filterbank.jit.hh"

namespace Blade::Modules::Filterbank {

template<typename InputType>
Writer<InputType>::Writer(const Config& config, const Input& input) 
        : Module(filterbank_program),
          config(config),
          input(input) {


    const auto inputDims = this->input.buffer.dims();
    if (config.beamNames.size() != inputDims.numberOfAspects()) {
        BL_FATAL(
            "Number of beam names does not match the number of beams: {} != {}",
            config.beamNames.size(), inputDims.numberOfAspects()
        );
        BL_CHECK_THROW(Result::ASSERTION_ERROR);
    }
    if (config.beamCoordinates.size() != inputDims.numberOfAspects()) {
        BL_FATAL(
            "Number of beam co-ordinates does not match the number of beams: {} != {}",
            config.beamCoordinates.size(), inputDims.numberOfAspects()
        );
        BL_CHECK_THROW(Result::ASSERTION_ERROR);
    }

    this->fileDescriptors.resize(inputDims.numberOfAspects());

    this->filterbank_header.machine_id = this->config.machineId;
    this->filterbank_header.telescope_id = filterbank_telescope_id(this->config.telescopeName.c_str());
    this->filterbank_header.data_type = 1;
    this->filterbank_header.barycentric = this->config.baryCentric;
    this->filterbank_header.pulsarcentric = this->config.pulsarCentric;
    this->filterbank_header.az_start = this->config.azimuthStart;
    this->filterbank_header.za_start = this->config.zenithStart;
    this->filterbank_header.fch1 = 1e-6 * (this->config.firstChannelFrequencyHz);
    this->filterbank_header.foff = 1e-6 * (this->config.bandwidthHz / inputDims.numberOfFrequencyChannels());
    this->filterbank_header.nchans = (I32) inputDims.numberOfFrequencyChannels();
    this->filterbank_header.nbeams = (I32) inputDims.numberOfAspects();
    this->filterbank_header.ibeam = -1;
    this->filterbank_header.nbits = (I32) sizeof(InputType)*8;
    this->filterbank_header.tstart = this->config.julianDateStart - 2400000.5; // from JD to MJD
    this->filterbank_header.tsamp = abs(1.0/(this->config.bandwidthHz / inputDims.numberOfFrequencyChannels())); // time always moves forward
    this->filterbank_header.nifs = this->config.numberOfIfChannels;
    strncpy(this->filterbank_header.rawdatafile, this->config.sourceDataFilename.c_str(), 79);

    this->openFilesWriteHeaders();

    BL_DEBUG("Recorded Bandwidth: {}", this->config.bandwidthHz);
    BL_DEBUG("Fch1: {}", this->filterbank_header.fch1);
    BL_DEBUG("Fch center: {}", this->filterbank_header.fch1 + this->filterbank_header.foff*this->filterbank_header.nchans/2);

    // Print configuration buffers.
    BL_INFO("Output File Path: {}", config.filepath);
    BL_INFO("Frequency Batches per Input: {}", config.numberOfInputFrequencyChannelBatches);
    BL_INFO("Data Dimensions [B, F, T, P]: {} -> {}", inputDims, "N/A");
}

template<typename InputType>
void Writer<InputType>::openFilesWriteHeaders() {
    // Check configuration values.
    for (size_t i = 0; i < this->fileDescriptors.size(); i++) {
        auto filepath = fmt::format("{}-beam{:04}.fil", this->config.filepath, i % 10000);
        this->fileDescriptors[i] = open(filepath.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (this->fileDescriptors[i] < 1) {
            BL_FATAL("Could not open '{}': {}\n", filepath, this->fileDescriptors[i]);
            BL_CHECK_THROW(Result::ERROR);
        }
        this->filterbank_header.ibeam = i;

        this->filterbank_header.src_raj = this->config.beamCoordinates[i].RA * 12.0 / BL_PHYSICAL_CONSTANT_PI; // from radians to hours
        this->filterbank_header.src_dej = this->config.beamCoordinates[i].DEC * 180.0 / BL_PHYSICAL_CONSTANT_PI;
        strncpy(this->filterbank_header.source_name, this->config.beamNames[i].c_str(), 79);

        filterbank_fd_write_header(this->fileDescriptors[i], &this->filterbank_header);
    }
}

template<typename InputType>
const Result Writer<InputType>::preprocess(const cudaStream_t& stream, const U64& currentComputeCount) {
    // Expect data with dimensions: [Fbatch, B, T, P, F]

    const auto inputDims = this->input.buffer.dims();
    const U64 numberOfTimePolarizationSamples = inputDims.numberOfTimeSamples()*inputDims.numberOfPolarizations();
    const U64 frequencyBatchByteStride = this->input.buffer.size_bytes()/this->config.numberOfInputFrequencyChannelBatches;
    const U64 aspectByteStride = frequencyBatchByteStride/inputDims.numberOfAspects();
    const U64 timepolSampleByteStride = (inputDims.numberOfFrequencyChannels()/this->config.numberOfInputFrequencyChannelBatches)*sizeof(InputType);

    U64 bytesWritten = 0;
    const long max_iovecs = sysconf(_SC_IOV_MAX);
    
    struct iovec* iovecs = (struct iovec*) malloc(max_iovecs * sizeof(struct iovec));
    int iovec_count = 0;

    for (size_t a = 0; a < inputDims.numberOfAspects(); a++) {
        if (this->config.numberOfInputFrequencyChannelBatches == 1) {
            bytesWritten += write(
                this->fileDescriptors[a],
                this->input.buffer.data() + a*aspectByteStride/sizeof(InputType),
                aspectByteStride
            );
        } else {
            for (size_t tp = 0; tp < numberOfTimePolarizationSamples; tp++) {
                for (size_t fb = 0; fb < this->config.numberOfInputFrequencyChannelBatches; fb++) {
                    iovecs[iovec_count].iov_base = this->input.buffer.data() +
                        ( fb * frequencyBatchByteStride
                        + a * aspectByteStride
                        + tp * timepolSampleByteStride
                        )/sizeof(InputType);
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
