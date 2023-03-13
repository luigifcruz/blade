#define BL_LOG_DOMAIN "M::CHANNELIZER"

#include "blade/modules/channelizer/base.hh"

#include "channelizer.jit.hh"

namespace Blade::Modules {

template<typename IT, typename OT>
Channelizer<IT, OT>::Channelizer(const Config& config, 
                                 const Input& input,
                                 const cudaStream_t& stream)
        : Module(channelizer_program),
          config(config),
          input(input),
          post_block(config.blockSize) {
    // Check configuration values.
    if ((getInputBuffer().numberOfTimeSamples() % config.rate) != 0) {
        BL_FATAL("The number of time samples ({}) should be divisable "
                "by the channelizer rate ({}).", getInputBuffer().numberOfTimeSamples(),
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

    // Link output buffer or link input with output.
    BL_CHECK_THROW(Memory::Link(output.buf, input.buf, getOutputBufferShape()));

    // Print configuration values.
    BL_INFO("Type: {} -> {}", TypeInfo<IT>::name, TypeInfo<OT>::name);
    BL_INFO("Shape [A, F, T, P]: {} -> {}", getInputBuffer().shape(), getOutputBuffer().shape());
    BL_INFO("FFT Size: {}", config.rate);
    BL_INFO("FFT Backend: cuFFT Callbacks");

    // Check FFT rate.
    if (config.rate == 1) {
        return;
    }

    if (config.rate != getInputBuffer().numberOfTimeSamples()) {
        BL_FATAL("Due to performance reasons, channelization with rates "
                 "different than the number of time samples are not "
                 "supported anymore.");
        BL_CHECK_THROW(Result::ERROR);
    }

    // FFT dimension (1D, 2D, ...).
    int rank = 1;

    // FFT size for each dimension.
    int n[] = { static_cast<int>(config.rate) }; 

    // Distance between successive input element and output element.
    int istride = getInputBuffer().numberOfPolarizations();
    int ostride = getInputBuffer().numberOfPolarizations();

    // Distance between input batches and output batches.
    int idist = (config.rate * getInputBuffer().numberOfPolarizations());
    int odist = (config.rate * getInputBuffer().numberOfPolarizations());

    // Input size with pitch, this is ignored for 1D tansformations.
    int inembed[] = { 0 }; 
    int onembed[] = { 0 };

    // Number of batched FFTs.
    int batch = (getInputBuffer().size() / getInputBuffer().numberOfPolarizations()) / config.rate;

    // Create cuFFT plan.
    cufftCreate(&plan);
    cufftPlanMany(&plan, rank, n,
                  inembed, istride, idist,
                  onembed, ostride, odist,
                  CUFFT_C2C, batch);
    cufftSetStream(plan, stream);

    // Install callbacks.
    callback = std::make_unique<Internal::Callback>(plan, input.buf.numberOfPolarizations());
}

template<typename IT, typename OT>
Channelizer<IT, OT>::~Channelizer() {
    if (config.rate != 1) {
        cufftDestroy(plan);
    }
}

template<typename IT, typename OT>
const Result Channelizer<IT, OT>::process(const cudaStream_t& stream) {
    if (config.rate == 1) {
        return Result::SUCCESS;
    } 

    cufftComplex* input_ptr = reinterpret_cast<cufftComplex*>(input.buf.data()); 
    cufftComplex* output_ptr = reinterpret_cast<cufftComplex*>(output.buf.data()); 

    for (U64 pol = 0; pol < getInputBuffer().numberOfPolarizations(); pol++) {
        BL_CUFFT_CHECK(cufftExecC2C(plan, input_ptr + pol, output_ptr + pol, CUFFT_FORWARD), [&]{
            BL_FATAL("cuFFT failed to execute: {0:#x}", err);
        });
    }

    BL_CUDA_CHECK_KERNEL([&]{
        BL_FATAL("Module failed to execute: {}", err);
        return Result::CUDA_ERROR;
    });

    return Result::SUCCESS;
}

template class BLADE_API Channelizer<CF32, CF32>;

}  // namespace Blade::Modules
