#include "blade/modules/channelizer.hh"

#include "channelizer.jit.hh"

namespace Blade::Modules {

// TODO: Implement multiple beams capability;

template<typename IT, typename OT>
Result Channelizer<IT, OT>::initializeCufft() {
    // FFT dimension (1D, 2D, ...)
    int rank = 1;

    // FFT size for each dimension
    int n[] = { static_cast<int>(config.rate) }; 

    // Distance between successive input element and output element.
    int istride = config.numberOfPolarizations;
    int ostride = config.numberOfPolarizations;

    // Distance between input batches and output batches.
    int idist = (config.rate * config.numberOfPolarizations);
    int odist = (config.rate * config.numberOfPolarizations);

    // Input size with pitch, this is ignored for 1D tansformations.
    int inembed[] = { 0 }; 
    int onembed[] = { 0 };

    // Number of batched FFTs.
    int batch = (getBufferSize() / config.numberOfPolarizations) / config.rate; 

    // Create cuFFT plan.
    cufftPlanMany(&plan, rank, n, 
                  inembed, istride, idist,
                  onembed, ostride, odist,
                  CUFFT_C2C, batch);

    return Result::SUCCESS;
}

template<typename IT, typename OT>
Result Channelizer<IT, OT>::initializeInternal() {
    std::string kernel_key;
    switch (config.rate) {
        case 4: kernel_key = "fft_4pnt"; break;
        default:
            BL_FATAL("The channelize rate of {} is not supported yet.", config.rate);
            throw Result::ERROR;
    }

    grid = 
        (
            (
                getBufferSize() / 
                config.rate /
                config.numberOfPolarizations
            ) + block.x - 1
        ) / block.x;

    kernel =
        Template(kernel_key)
            .instantiate(getBufferSize(),
                         config.rate,
                         config.numberOfPolarizations,
                         config.numberOfTimeSamples,
                         config.numberOfFrequencyChannels);

    return Result::SUCCESS;
}

template<typename IT, typename OT>
Channelizer<IT, OT>::Channelizer(const Config& config, const Input& input)
        : Module(config.blockSize, channelizer_kernel),
          config(config),
          input(input) {
    BL_INFO("===== Channelizer Module Configuration");

    if ((config.numberOfTimeSamples % config.rate) != 0) {
        BL_FATAL("The number of time samples ({}) should be divisable "
                "by the channelizer rate ({}).", config.numberOfTimeSamples,
                config.rate);
        throw Result::ERROR;
    }

    if (config.numberOfBeams != 1) {
        BL_WARN("Number of beams ({}) should be one.", config.numberOfBeams);
        throw Result::ERROR;
    }

    if (config.rate == config.numberOfTimeSamples) {
        BL_INFO("FFT Backend: cuFFT");
        BL_CHECK_THROW(initializeCufft());
    } else {
        if (config.rate == 1) {
            BL_INFO("FFT Backend: Bypass");
            BL_CHECK_THROW(output.buf.link(input.buf));
        } else {
            BL_INFO("FFT Backend: Internal");
            BL_CHECK_THROW(initializeInternal());
        }
    } 

    BL_INFO("Number of Beams: {}", config.numberOfBeams);
    BL_INFO("Number of Antennas: {}", config.numberOfAntennas);
    BL_INFO("Number of Frequency Channels: {}", config.numberOfFrequencyChannels);
    BL_INFO("Number of Time Samples: {}", config.numberOfTimeSamples);
    BL_INFO("Number of Polarizations: {}", config.numberOfPolarizations);
    BL_INFO("Channelizer Rate: {}", config.rate);

    BL_CHECK_THROW(InitInput(input.buf, getBufferSize()));
    BL_CHECK_THROW(InitOutput(output.buf, getBufferSize()));
}

template<typename IT, typename OT>
Result Channelizer<IT, OT>::process(const cudaStream_t& stream) {
    if (config.rate == config.numberOfTimeSamples) {
        cufftSetStream(plan, stream);
        for (U64 pol = 0; pol < config.numberOfPolarizations; pol++) {
            cufftComplex* input_ptr = reinterpret_cast<cufftComplex*>(input.buf.data()); 
            cufftComplex* output_ptr = reinterpret_cast<cufftComplex*>(output.buf.data()); 
            cufftExecC2C(plan, input_ptr + pol, output_ptr + pol, CUFFT_FORWARD);
        }
    } else {
        if (config.rate == 1) {
            return Result::SUCCESS;
        } else {
            cache
                .get_kernel(kernel)
                ->configure(grid, block, 0, stream)
                ->launch(input.buf.data(), output.buf.data());
        }
    } 

    BL_CUDA_CHECK_KERNEL([&]{
        BL_FATAL("Module failed to execute: {}", err);
        return Result::CUDA_ERROR;
    });

    return Result::SUCCESS;
}

template class BLADE_API Channelizer<CF32, CF32>;

}  // namespace Blade::Modules
