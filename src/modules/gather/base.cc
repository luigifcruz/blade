#define BL_LOG_DOMAIN "M::GATHER"

#include <type_traits>
#include <typeindex>

#include "blade/modules/gather.hh"

#include "gather.jit.hh"

namespace Blade::Modules {

template<typename IT, typename OT>
Gather<IT, OT>::Gather(const Config& config,
                       const Input& input,
                       const cudaStream_t& stream)
        : Module(gather_program),
          config(config),
          input(input),
          computeRatio(config.multiplier) {
    if constexpr (std::is_same<IT, OT>::value) {
        BL_FATAL("Input ({}) and output ({}) types aren't the same. Casting isn't supported by Gather yet.",
                 TypeInfo<IT>::name, TypeInfo<OT>::name);
        BL_CHECK_THROW(Result::ERROR);
    }

    if (config.axis >= input.buf.shape().dimensions()) {
        BL_FATAL("Selected input axis ({}) is larger than input shape dimensions ({}).",
                 config.axis, input.buf.shape().dimensions());
        BL_CHECK_THROW(Result::ERROR);
    }
    
    const auto choppedShape = ArrayShape::Type(input.buf.shape()) | std::views::drop(config.axis);
    const U64 copySize = std::accumulate(choppedShape.begin(), choppedShape.end(), 0);
    BL_DEBUG("Copy size of {} elements.", copySize);

    if (copySize < config.copySizeThreshold) {
        strategy = Strategy::Kernel;
    } else {
        strategy = Strategy::Copy;
    }

    // Allocate output buffers.
    output.buf = ArrayTensor<Device::CUDA, OT>(getOutputBufferShape());

    // Print configuration values.
    BL_INFO("Type: {} -> {}", TypeInfo<IT>::name, TypeInfo<OT>::name);
    BL_INFO("Shape: {} -> {}", getInputBuffer().shape(), 
                               getOutputBuffer().shape());
    BL_INFO("Axis: {}", config.axis);
    BL_INFO("Multiplier: {}", config.multiplier);
    BL_INFO("Strategy: {}", (strategy == Strategy::Kernel) ? "KERNEL" : "COPY");
}

template<typename IT, typename OT>
Result Gather<IT, OT>::process(const cudaStream_t& stream, const U64& currentStepCount) {
    if (strategy == Strategy::Kernel) {
        cache
            .get_kernel(
                Template("accumulate")
                    .instantiate(
                        TypeInfo<IT>::name,
                        config.axis,
                        currentStepCount
                    )
            )
            ->configure(
                PadGridSize(input.buf.size(), config.blockSize),
                config.blockSize,
                0,
                stream
            )
            ->launch(input.buf, output.buf);
    }

    if (strategy == Strategy::Copy) {
        const auto& inputHeight = output.buf.shape().numberOfAspects() * 
                                  output.buf.shape().numberOfFrequencyChannels();
        const auto& inputWidth = input.buf.size_bytes() / inputHeight;
        const auto& outputPitch = inputWidth * config.multiplier;

        BL_CHECK(
            Copy2D(
                output.buf,
                outputPitch,
                0 * inputWidth,

                input.buf,
                inputWidth,
                0,

                inputWidth,
                inputHeight, 
                stream
            )
        );
    }

    return Result::SUCCESS;
}

template class BLADE_API Gather<CI8, CI8>;
template class BLADE_API Gather<CF16, CF16>;
template class BLADE_API Gather<CF32, CF32>;

}  // namespace Blade::Modules
