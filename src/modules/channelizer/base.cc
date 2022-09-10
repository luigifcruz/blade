#define BL_LOG_DOMAIN "M::CHANNELIZER"

#include "blade/modules/channelizer.hh"

#include "channelizer.jit.hh"

namespace Blade::Modules {

template<typename IT, typename OT>
Channelizer<IT, OT>::Channelizer(const Config& config, const Input& input)
        : Module(config.blockSize, channelizer_kernel),
          config(config),
          input(input),
          pre_block(config.blockSize),
          post_block(config.blockSize) {
    // Check configuration values.
    if ((getInputBuffer().dims().numberOfTimeSamples() % config.rate) != 0) {
        BL_FATAL("The number of time samples ({}) should be divisable "
                "by the channelizer rate ({}).", getInputBuffer().dims().numberOfTimeSamples(),
                config.rate);
        BL_CHECK_THROW(Result::ERROR);
    }

    if (((config.rate % 2) != 0) && (config.rate != 1)) {
        BL_FATAL("The channelizer rate ({}) should be divisable by 2.", config.rate);
        throw Result::ERROR;
    }

    if (((config.rate % 2) != 0) && (config.rate != 1)) {
        BL_FATAL("The channelizer rate ({}) should be divisable by 2.", config.rate);
        throw Result::ERROR;
    }

    // Allocate output buffers.
    BL_CHECK_THROW(output.buf.resize(getOutputBufferDims()));

    // Print configuration values.
    BL_INFO("Type: {} -> {}", TypeInfo<IT>::name, TypeInfo<OT>::name);
    BL_INFO("Dimensions [A, F, T, P]: {} -> {}", getInputBuffer().dims(), getOutputBuffer().dims());
    BL_INFO("FFT Size: {}", config.rate);

    // Configure FFT chain.
    if (config.rate == 1) {
        BL_INFO("FFT Backend: Bypass");
        BL_CHECK_THROW(output.buf.link(input.buf));
        return;
    }

    switch (config.rate) {
        case 4:  kernel_key = "fft_4pnt"; break;
        default: kernel_key = "cufft"; break;
    }

    if (kernel_key == "cufft") {
        BL_INFO("FFT Backend: cuFFT");

        // FFT dimension (1D, 2D, ...)
        int rank = 1;

        // FFT size for each dimension
        int n[] = { static_cast<int>(config.rate) }; 

        // Distance between successive input element and output element.
        int istride = getInputBuffer().dims().numberOfPolarizations();
        int ostride = getInputBuffer().dims().numberOfPolarizations();

        // Distance between input batches and output batches.
        int idist = (config.rate * getInputBuffer().dims().numberOfPolarizations());
        int odist = (config.rate * getInputBuffer().dims().numberOfPolarizations());

        // Input size with pitch, this is ignored for 1D tansformations.
        int inembed[] = { 0 }; 
        int onembed[] = { 0 };

        // Number of batched FFTs.
        int batch = (getInputBuffer().size() / getInputBuffer().dims().numberOfPolarizations()) / config.rate; 

        // Create cuFFT plan.
        cufftPlanMany(&plan, rank, n, 
                      inembed, istride, idist,
                      onembed, ostride, odist,
                      CUFFT_C2C, batch);

        // Perform FFT shift before cuFFT.
        pre_kernel = Template("shifter").instantiate(getInputBuffer().size(), getInputBuffer().dims().numberOfPolarizations());
        pre_grid = dim3((getInputBuffer().size() + pre_block.x - 1) / pre_block.x);

        if (config.rate != getInputBuffer().dims().numberOfTimeSamples()) {
            BL_CHECK_THROW(buffer.resize(getInputBuffer().dims()));
            BL_CHECK_THROW(indices.resize(getInputBuffer().dims()));

            // Generate post-FFT indices.
            // This really should be calculated on the GPU, 
            // but this is faster to write and it probably
            // won't be used much. Please rewrite this if
            // used regurlarly.
            U64 i = 0;
            U64 numberOfAspects = getInputBuffer().dims().numberOfAspects();
            U64 numberOfFrequencyChannels = getInputBuffer().dims().numberOfFrequencyChannels();
            U64 numberOfTimeSamples = getInputBuffer().dims().numberOfTimeSamples();
            U64 numberOfPolarizations = getInputBuffer().dims().numberOfPolarizations();

            for (U64 a = 0; a < numberOfAspects; a++) {
                const U64 a_off = a * 
                                  numberOfFrequencyChannels * 
                                  numberOfTimeSamples * 
                                  numberOfPolarizations; 

                for (U64 c = 0; c < numberOfFrequencyChannels; c++) {
                    const U64 c_off = c * 
                                      numberOfTimeSamples * 
                                      numberOfPolarizations; 

                    for (U64 r = 0; r < config.rate; r++) {
                        const U64 r_off = r * numberOfPolarizations;

                        for (U64 o = 0; o < (numberOfTimeSamples / config.rate); o++) {
                            const U64 o_off = o * config.rate * numberOfPolarizations;

                            for (U64 p = 0; p < numberOfPolarizations; p++) {
                                indices[i++] = a_off + c_off + o_off + r_off + p;
                            }
                        }
                    }
                }
            }

            post_kernel = Template("shuffler").instantiate(getInputBuffer().size());
            post_grid = dim3((getInputBuffer().size() + post_block.x - 1) / post_block.x);
        }
    } else {
        BL_INFO("FFT Backend: Internal");

        grid = 
            (
                (
                    getInputBuffer().size() / 
                    config.rate /
                    getInputBuffer().dims().numberOfPolarizations()
                ) + block.x - 1
            ) / block.x;

        kernel =
            Template(kernel_key)
                .instantiate(getInputBuffer().size(),
                             config.rate,
                             getInputBuffer().dims().numberOfPolarizations(),
                             getInputBuffer().dims().numberOfTimeSamples(),
                             getInputBuffer().dims().numberOfFrequencyChannels());
    }
}

template<typename IT, typename OT>
const Result Channelizer<IT, OT>::process(const cudaStream_t& stream) {
    if (config.rate == 1) {
        return Result::SUCCESS;
    } 

    if (kernel_key == "cufft") {
        cufftSetStream(plan, stream);

        cache
            .get_kernel(pre_kernel)
            ->configure(pre_grid, pre_block, 0, stream)
            ->launch(input.buf.data(), output.buf.data());

        for (U64 pol = 0; pol < getInputBuffer().dims().numberOfPolarizations(); pol++) {
            cufftComplex* input_ptr = reinterpret_cast<cufftComplex*>(output.buf.data()); 
            cufftComplex* output_ptr = reinterpret_cast<cufftComplex*>(buffer.data()); 

            if (config.rate == getInputBuffer().dims().numberOfTimeSamples()) {
                output_ptr = reinterpret_cast<cufftComplex*>(output.buf.data());
            }

            cufftExecC2C(plan, input_ptr + pol, output_ptr + pol, CUFFT_FORWARD);
        }

        if (config.rate != getInputBuffer().dims().numberOfTimeSamples()) {
            cache
                .get_kernel(post_kernel)
                    ->configure(post_grid, post_block, 0, stream)
                    ->launch(buffer.data(), indices.data(), output.buf.data());
        }
    } else {
        cache
            .get_kernel(kernel)
                ->configure(grid, block, 0, stream)
                ->launch(input.buf.data(), output.buf.data());
    }

    BL_CUDA_CHECK_KERNEL([&]{
        BL_FATAL("Module failed to execute: {}", err);
        return Result::CUDA_ERROR;
    });

    return Result::SUCCESS;
}

template class BLADE_API Channelizer<CF32, CF32>;

}  // namespace Blade::Modules
