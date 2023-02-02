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
            TypeInfo<IT>::cudaName,
            TypeInfo<OT>::cudaName,
            getInputBuffer().size() / 2
        )
    );

    if constexpr (!std::is_same<IT, OT>::value) {
        BL_FATAL("This module requires the type of the input "
                 "({}) and output ({}) to be the same.",
                 TypeInfo<IT>::name, TypeInfo<OT>::name); 
        BL_INFO("Contact the maintainer if this"
                "functionality is required.");
        BL_CHECK_THROW(Result::ERROR);
    }

    // Link output buffers.
    if (config.mode == Mode::BYPASS) {
        BL_INFO("Bypass: Enabled");
    }
    BL_CHECK_THROW(output.buf.link(input.buf));

    // Print configuration values.
    BL_INFO("Type: {} -> {}", TypeInfo<IT>::name, TypeInfo<OT>::name);
    BL_INFO("Dimensions [A, F, T, P]: {} -> {}", getInputBuffer().dims(), 
                                                 getOutputBuffer().dims());
}

template<typename IT, typename OT>
const Result Polarizer<IT, OT>::process(const cudaStream_t& stream) {
    if (config.mode == Mode::BYPASS) {
        return Result::SUCCESS;
    }

    return this->runKernel("main", stream, input.buf.data(), output.buf.data());
}

template class BLADE_API Polarizer<CF32, CF32>;

}  // namespace Blade::Modules
