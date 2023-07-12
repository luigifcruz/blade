#define BL_LOG_DOMAIN "M::POLARIZER"

#include <type_traits>
#include <typeindex>

#include "blade/modules/polarizer.hh"

#include "polarizer.jit.hh"

namespace Blade::Modules {

template<typename IT, typename OT>
Polarizer<IT, OT>::Polarizer(const Config& config, 
                             const Input& input, 
                             const cudaStream_t& stream)
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
            TypeInfo<IT>::name,
            TypeInfo<OT>::name
        )
    );

    if constexpr (!std::is_same<IT, OT>::value) {
        BL_FATAL("This module requires the type of the input "
                 "({}) and output ({}) to be the same.",
                 TypeInfo<IT>::name, TypeInfo<OT>::name); 
        BL_INFO("Contact the maintainer if this "
                "functionality is required.");
        BL_CHECK_THROW(Result::ERROR);
    }

    // Link output buffers.
    if (config.mode == Mode::BYPASS) {
        BL_INFO("Bypass: Enabled");
    }

    // Link output buffer or link input with output.
    BL_CHECK_THROW(Memory::Link(output.buf, input.buf));

    // Print configuration values.
    BL_INFO("Type: {} -> {}", TypeInfo<IT>::name, TypeInfo<OT>::name);
    BL_INFO("Shape: {} -> {}", getInputBuffer().shape(), 
                               getOutputBuffer().shape());
}

template<typename IT, typename OT>
Result Polarizer<IT, OT>::process(const cudaStream_t& stream, const U64& currentStepNumber) {
    if (config.mode == Mode::BYPASS) {
        return Result::SUCCESS;
    }

    return this->runKernel("main", stream, input.buf, output.buf);
}

template class BLADE_API Polarizer<CF32, CF32>;
template class BLADE_API Polarizer<CF16, CF16>;

}  // namespace Blade::Modules
