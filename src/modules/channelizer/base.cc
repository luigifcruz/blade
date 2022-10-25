#define BL_LOG_DOMAIN "M::CHANNELIZER"

#include "blade/modules/channelizer.hh"

#include "channelizer.jit.hh"

namespace Blade::Modules {

template<typename IT, typename OT>
Channelizer<IT, OT>::Channelizer(const Config& config, const Input& input)
        : Module(channelizer_program),
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

    // Allocate output buffer or link input with output.
    if (config.rate == 1) {
        BL_CHECK_THROW(output.buf.link(input.buf));
    } else {
        BL_CHECK_THROW(output.buf.resize(getOutputBufferDims()));
    }

    // Print configuration values.
    BL_INFO("Type: {} -> {}", TypeInfo<IT>::name, TypeInfo<OT>::name);
    BL_INFO("Dimensions [A, F, T, P]: {} -> {}", getInputBuffer().dims(), getOutputBuffer().dims());
    BL_INFO("FFT Size: {}", config.rate);

    if (config.rate == 1) {
        BL_INFO("FFT Backend: Bypass");
        return;
    }

    // Configure FFT chain.
    if (config.rate != 4) {
        BL_INFO("FFT Backend: cuFFT");

        // FFT dimension (1D, 2D, ...).
        int rank = 1;

        // FFT size for each dimension.
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
        BL_CHECK_THROW(
            createKernel(
                // Kernel name.
                "pre",
                // Kernel function key.
                "shifter",
                // Kernel grid & block size.
                PadGridSize(getInputBuffer().size(), config.blockSize), 
                config.blockSize,
                // Kernel templates.
                getInputBuffer().size(),
                getInputBuffer().dims().numberOfPolarizations()
            )
        );

        if (config.rate != getInputBuffer().dims().numberOfTimeSamples()) {
            BL_WARN("Using a slow FFT implementation because channelization "
                    "rate is different than number of time samples.");

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

            BL_CHECK_THROW(
                createKernel(
                    // Kernel name.
                    "post",
                    // Kernel function key.
                    "shuffler",
                    // Kernel grid & block size.
                    PadGridSize(getInputBuffer().size(), config.blockSize), 
                    config.blockSize,
                    // Kernel templates.
                    getInputBuffer().size()
                )
            );
        }
    } else {
        BL_INFO("FFT Backend: Internal");

        BL_CHECK_THROW(
            createKernel(
                // Kernel name.
                "internal",
                // Kernel function key.
                "fft_4pnt",
                // Kernel grid & block size.
                PadGridSize(
                    getInputBuffer().size() / 
                    config.rate /
                    getInputBuffer().dims().numberOfPolarizations(), config.blockSize),
                config.blockSize,
                // Kernel templates.
                getInputBuffer().size(),
                config.rate,
                getInputBuffer().dims().numberOfPolarizations(),
                getInputBuffer().dims().numberOfTimeSamples(),
                getInputBuffer().dims().numberOfFrequencyChannels()
            )
        );
    }
}

template<typename IT, typename OT>
Channelizer<IT, OT>::~Channelizer() {
    if (config.rate != 4 && config.rate != 1) {
        cufftDestroy(plan);
    }
}

template<typename IT, typename OT>
const Result Channelizer<IT, OT>::process(const cudaStream_t& stream) {
    if (config.rate == 1) {
        return Result::SUCCESS;
    } 

    if (config.rate != 4) {
        cufftSetStream(plan, stream);

        BL_CHECK(runKernel("pre", stream, input.buf.data(), output.buf.data()));

        for (U64 pol = 0; pol < getInputBuffer().dims().numberOfPolarizations(); pol++) {
            cufftComplex* input_ptr = reinterpret_cast<cufftComplex*>(output.buf.data()); 
            cufftComplex* output_ptr = reinterpret_cast<cufftComplex*>(buffer.data()); 

            if (config.rate == getInputBuffer().dims().numberOfTimeSamples()) {
                output_ptr = reinterpret_cast<cufftComplex*>(output.buf.data());
            }

            cufftExecC2C(plan, input_ptr + pol, output_ptr + pol, CUFFT_FORWARD);
        }

        if (config.rate != getInputBuffer().dims().numberOfTimeSamples()) {
            BL_CHECK(runKernel("post", stream, buffer.data(), indices.data(), output.buf.data()));
        }
    } else {
        BL_CHECK(runKernel("internal", stream, input.buf.data(), output.buf.data()));
    }

    return Result::SUCCESS;
}

template class BLADE_API Channelizer<CF32, CF32>;

}  // namespace Blade::Modules
