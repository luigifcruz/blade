#define BL_LOG_DOMAIN "M::POLARIZER"

#include <type_traits>
#include <typeindex>

#include "blade/modules/polarizer.hh"

#include "polarizer.jit.hh"

namespace Blade::Modules {

template<typename IT, typename OT>
Polarizer<IT, OT>::Polarizer(const Config& config, const Input& input)
        : Module(polarizer_program),
          config(config),
          input(input) {
    // Configure kernel instantiation.
    BL_CHECK_THROW(
        this->createKernel(
            // Kernel name.
            "main",
            // Kernel function key.
            "polarizer",
            // Kernel grid & block size.
            PadGridSize(
                getInputBuffer().size(), 
                config.blockSize
            ),
            config.blockSize,
            // Kernel templates.
            CudaType<IT>(),
            CudaType<OT>(),
            getInputBuffer().size()
        )
    );

    // Allocate output buffers.
    BL_CHECK_THROW(output.buf.resize(getOutputBufferDims()));

    // Print configuration values.
    BL_INFO("Type: {} -> {}", TypeInfo<IT>::name, TypeInfo<OT>::name);
    BL_INFO("Dimensions [A, F, T, P]: {} -> {}", getInputBuffer().dims(), 
                                                 getOutputBuffer().dims());
}

template<typename IT, typename OT>
const Result Polarizer<IT, OT>::process(const cudaStream_t& stream) {
    return this->runKernel("main", stream, input.buf.data(), output.buf.data());
}

template class BLADE_API Polarizer<CI8, CI8>;
template class BLADE_API Polarizer<CF16, CF16>;
template class BLADE_API Polarizer<CF32, CF32>;
template class BLADE_API Polarizer<CF64, CF64>;

}  // namespace Blade::Modules
