#include "blade/modules/channelizer.hh"


#include "channelizer.jit.hh"

namespace Blade::Modules {

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

    BL_INFO("Number of Beams: {}", config.numberOfBeams);
    BL_INFO("Number of Antennas: {}", config.numberOfAntennas);
    BL_INFO("Number of Frequency Channels: {}", config.numberOfFrequencyChannels);
    BL_INFO("Number of Time Samples: {}", config.numberOfTimeSamples);
    BL_INFO("Number of Polarizations: {}", config.numberOfPolarizations);
    BL_INFO("Channelizer Rate: {}", config.rate);

    if (config.rate == 1) {
        BL_INFO("FFT Backend: Bypass");
        BL_CHECK_THROW(output.buf.link(input.buf));
        return;
    }

    BL_INFO("FFT Backend: cuFFT");

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

    BL_CHECK_THROW(buffer.resize(getBufferSize()));
    BL_CHECK_THROW(indices.resize(getBufferSize()));

    // Generate post-FFT indices.
    U64 numberOfBeams = config.numberOfBeams;
    U64 numberOfFrequencyChannels = config.numberOfFrequencyChannels * config.rate;
    U64 numberOfTimeSamples = config.numberOfTimeSamples / config.rate;
    U64 numberOfPolarizations = config.numberOfPolarizations;

    for (U64 beam = 0; beam < numberOfBeams; beam++) {
        const U64 beam_off = beam * numberOfFrequencyChannels * numberOfTimeSamples * numberOfPolarizations;

        for (U64 ch = 0; ch < numberOfFrequencyChannels; ch++) {
            const U64 ch_off = ch * numberOfTimeSamples * numberOfPolarizations;
            const U64 ch_res = ch * numberOfPolarizations;

            for (U64 ts = 0; ts < numberOfTimeSamples; ts++) {
                const U64 ts_off = ts * numberOfPolarizations;
                const U64 ts_res = ts * numberOfPolarizations * numberOfFrequencyChannels;

                for (U64 pol = 0; pol < numberOfPolarizations; pol++) {
                    const U64 pol_off = pol;
                    const U64 pol_res = pol;

                    indices[beam_off + ch_off + ts_off + pol_off] = beam_off + ch_res + ts_res + pol_res;
                }
            }
        }
    }

    kernel = Template("shuffle").instantiate(getBufferSize());
    grid = dim3((getBufferSize() + block.x - 1) / block.x);

    BL_CHECK_THROW(InitInput(input.buf, getBufferSize()));
    BL_CHECK_THROW(InitOutput(output.buf, getBufferSize()));
}

template<typename IT, typename OT>
Result Channelizer<IT, OT>::process(const cudaStream_t& stream) {
    if (config.rate == 1) {
        return Result::SUCCESS;
    } 

    cufftSetStream(plan, stream);
    for (U64 pol = 0; pol < config.numberOfPolarizations; pol++) {
        cufftComplex* input_ptr = reinterpret_cast<cufftComplex*>(input.buf.data()); 
        cufftComplex* output_ptr = reinterpret_cast<cufftComplex*>(buffer.data()); 
        cufftExecC2C(plan, input_ptr + pol, output_ptr + pol, CUFFT_FORWARD);
    }

    if (config.rate != config.numberOfTimeSamples) {
        cache
            .get_kernel(kernel)
            ->configure(grid, block, 0, stream)
            ->launch(buffer.data(), indices.data(), output.buf.data());
    }

    BL_CUDA_CHECK_KERNEL([&]{
        BL_FATAL("Module failed to execute: {}", err);
        return Result::CUDA_ERROR;
    });

    return Result::SUCCESS;
}

template class BLADE_API Channelizer<CF32, CF32>;

}  // namespace Blade::Modules
