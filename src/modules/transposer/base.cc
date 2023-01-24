#define BL_LOG_DOMAIN "M::TRANSPOSER"

#include <type_traits>
#include <typeindex>

#include "blade/modules/transposer.hh"

#include "transposer.jit.hh"

namespace Blade::Modules {

template<Device Dev, typename ElementType, ArrayDimensionOrder InputOrder, ArrayDimensionOrder OutputOrder>
Transposer<Dev, ElementType, InputOrder, OutputOrder>::Transposer(const Config& config, const Input& input)
        : Module(transposer_program),
          config(config),
          input(input) {

    this->bypass = false;
    if (InputOrder == OutputOrder) {
        this->bypass = true;
    }
    else if (
        InputOrder == ArrayDimensionOrder::AFTP
        && OutputOrder == ArrayDimensionOrder::ATPF
    ) {
        const auto dimensions = getInputBuffer().dims();
        this->bypass = dimensions.numberOfFrequencyChannels() == 1;
        this->bypass |= dimensions.numberOfTimeSamples()*dimensions.numberOfPolarizations() == 1;
    }
    else if (
        InputOrder == ArrayDimensionOrder::AFTP
        && OutputOrder == ArrayDimensionOrder::ATPFrev
    ) {
        const auto dimensions = getInputBuffer().dims();
        this->bypass = dimensions.numberOfFrequencyChannels() == 1;
    }

    if (!this->bypass) {
        // Allocate output buffers.
        BL_CHECK_THROW(output.buf.resize(getInputBuffer().dims()));
    }

    // Print configuration values.
    BL_INFO("Dimension Order: {} -> {}", ArrayDimensionOrderName(InputOrder), ArrayDimensionOrderName(OutputOrder));
    BL_INFO("Dimensions [A, F, T, P]: {} -> {}", getInputBuffer().dims(), output.buf.dims());
    BL_INFO("Bypassed: {}", this->bypass ? "True" : "False");
}

template<Device Dev, typename ElementType, ArrayDimensionOrder InputOrder, ArrayDimensionOrder OutputOrder>
const Result Transposer<Dev, ElementType, InputOrder, OutputOrder>::process(const cudaStream_t& stream) {
    if (this->bypass) {
        return Result::SUCCESS;
    }

    if (InputOrder == ArrayDimensionOrder::AFTP
        && (OutputOrder == ArrayDimensionOrder::ATPF || OutputOrder == ArrayDimensionOrder::ATPFrev)
    ) {
        const auto dimensions = getInputBuffer().dims();

        const U64 inputByteStrideA = getInputBuffer().size_bytes()/dimensions.numberOfAspects();
        const U64 inputByteStrideF = inputByteStrideA/dimensions.numberOfFrequencyChannels();
        const U64 inputByteStrideT = inputByteStrideF/dimensions.numberOfTimeSamples();
        const U64 inputByteStrideP = inputByteStrideT/dimensions.numberOfPolarizations();

        const U64 outputByteStrideA = getInputBuffer().size_bytes()/dimensions.numberOfAspects();
        const U64 outputByteStrideT = outputByteStrideA/dimensions.numberOfTimeSamples();
        const U64 outputByteStrideP = outputByteStrideT/dimensions.numberOfPolarizations();
        const U64 outputByteStrideF = outputByteStrideP/dimensions.numberOfFrequencyChannels();

        U64 fOut = 0;
        const BOOL reverseFrequency = OutputOrder == ArrayDimensionOrder::ATPFrev;

        for (U64 a = 0; a < dimensions.numberOfAspects(); a++) {
            for (U64 f = 0; f < dimensions.numberOfFrequencyChannels(); f++) {
                fOut = reverseFrequency ? dimensions.numberOfFrequencyChannels()-1-f : f;
                BL_CHECK(
                    Memory::Copy2D(
                        this->output.buf,
                        outputByteStrideP,
                        a*outputByteStrideA + fOut*outputByteStrideF, // dstOffset

                        this->input.buf,
                        inputByteStrideP,
                        a*inputByteStrideA + f*inputByteStrideF, // srcOffset

                        inputByteStrideP, // width
                        dimensions.numberOfTimeSamples()*dimensions.numberOfPolarizations(), // height
                        stream
                    )
                )
            }
        }

        return Result::SUCCESS;
    }

    return Result::ERROR;
}

// template class BLADE_API Transposer<Device::CUDA, F16,  ArrayDimensionOrder::AFTP, ArrayDimensionOrder::ATPFrev>;
template class BLADE_API Transposer<Device::CUDA, F32,  ArrayDimensionOrder::AFTP, ArrayDimensionOrder::ATPFrev>;
// template class BLADE_API Transposer<Device::CUDA, F64,  ArrayDimensionOrder::AFTP, ArrayDimensionOrder::ATPFrev>;
// template class BLADE_API Transposer<Device::CUDA, I8,   ArrayDimensionOrder::AFTP, ArrayDimensionOrder::ATPFrev>;
// template class BLADE_API Transposer<Device::CUDA, I16,  ArrayDimensionOrder::AFTP, ArrayDimensionOrder::ATPFrev>;
// template class BLADE_API Transposer<Device::CUDA, I32,  ArrayDimensionOrder::AFTP, ArrayDimensionOrder::ATPFrev>;
// template class BLADE_API Transposer<Device::CUDA, I64,  ArrayDimensionOrder::AFTP, ArrayDimensionOrder::ATPFrev>;
// template class BLADE_API Transposer<Device::CUDA, U8,   ArrayDimensionOrder::AFTP, ArrayDimensionOrder::ATPFrev>;
// template class BLADE_API Transposer<Device::CUDA, U16,  ArrayDimensionOrder::AFTP, ArrayDimensionOrder::ATPFrev>;
// template class BLADE_API Transposer<Device::CUDA, U32,  ArrayDimensionOrder::AFTP, ArrayDimensionOrder::ATPFrev>;
// template class BLADE_API Transposer<Device::CUDA, U64,  ArrayDimensionOrder::AFTP, ArrayDimensionOrder::ATPFrev>;
// template class BLADE_API Transposer<Device::CUDA, CF16, ArrayDimensionOrder::AFTP, ArrayDimensionOrder::ATPFrev>;
// template class BLADE_API Transposer<Device::CUDA, CF32, ArrayDimensionOrder::AFTP, ArrayDimensionOrder::ATPFrev>;
// template class BLADE_API Transposer<Device::CUDA, CF64, ArrayDimensionOrder::AFTP, ArrayDimensionOrder::ATPFrev>;
// template class BLADE_API Transposer<Device::CUDA, CI8,  ArrayDimensionOrder::AFTP, ArrayDimensionOrder::ATPFrev>;
// template class BLADE_API Transposer<Device::CUDA, CI16, ArrayDimensionOrder::AFTP, ArrayDimensionOrder::ATPFrev>;
// template class BLADE_API Transposer<Device::CUDA, CI32, ArrayDimensionOrder::AFTP, ArrayDimensionOrder::ATPFrev>;
// template class BLADE_API Transposer<Device::CUDA, CI64, ArrayDimensionOrder::AFTP, ArrayDimensionOrder::ATPFrev>;
// template class BLADE_API Transposer<Device::CUDA, CU8,  ArrayDimensionOrder::AFTP, ArrayDimensionOrder::ATPFrev>;
// template class BLADE_API Transposer<Device::CUDA, CU16, ArrayDimensionOrder::AFTP, ArrayDimensionOrder::ATPFrev>;
// template class BLADE_API Transposer<Device::CUDA, CU32, ArrayDimensionOrder::AFTP, ArrayDimensionOrder::ATPFrev>;
// template class BLADE_API Transposer<Device::CUDA, CU64, ArrayDimensionOrder::AFTP, ArrayDimensionOrder::ATPFrev>;

// template class BLADE_API Transposer<Device::CUDA, F16,  ArrayDimensionOrder::AFTP, ArrayDimensionOrder::ATPF>;
template class BLADE_API Transposer<Device::CUDA, F32,  ArrayDimensionOrder::AFTP, ArrayDimensionOrder::ATPF>;
// template class BLADE_API Transposer<Device::CUDA, F64,  ArrayDimensionOrder::AFTP, ArrayDimensionOrder::ATPF>;
// template class BLADE_API Transposer<Device::CUDA, I8,   ArrayDimensionOrder::AFTP, ArrayDimensionOrder::ATPF>;
// template class BLADE_API Transposer<Device::CUDA, I16,  ArrayDimensionOrder::AFTP, ArrayDimensionOrder::ATPF>;
// template class BLADE_API Transposer<Device::CUDA, I32,  ArrayDimensionOrder::AFTP, ArrayDimensionOrder::ATPF>;
// template class BLADE_API Transposer<Device::CUDA, I64,  ArrayDimensionOrder::AFTP, ArrayDimensionOrder::ATPF>;
// template class BLADE_API Transposer<Device::CUDA, U8,   ArrayDimensionOrder::AFTP, ArrayDimensionOrder::ATPF>;
// template class BLADE_API Transposer<Device::CUDA, U16,  ArrayDimensionOrder::AFTP, ArrayDimensionOrder::ATPF>;
// template class BLADE_API Transposer<Device::CUDA, U32,  ArrayDimensionOrder::AFTP, ArrayDimensionOrder::ATPF>;
// template class BLADE_API Transposer<Device::CUDA, U64,  ArrayDimensionOrder::AFTP, ArrayDimensionOrder::ATPF>;
// template class BLADE_API Transposer<Device::CUDA, CF16, ArrayDimensionOrder::AFTP, ArrayDimensionOrder::ATPF>;
// template class BLADE_API Transposer<Device::CUDA, CF32, ArrayDimensionOrder::AFTP, ArrayDimensionOrder::ATPF>;
// template class BLADE_API Transposer<Device::CUDA, CF64, ArrayDimensionOrder::AFTP, ArrayDimensionOrder::ATPF>;
// template class BLADE_API Transposer<Device::CUDA, CI8,  ArrayDimensionOrder::AFTP, ArrayDimensionOrder::ATPF>;
// template class BLADE_API Transposer<Device::CUDA, CI16, ArrayDimensionOrder::AFTP, ArrayDimensionOrder::ATPF>;
// template class BLADE_API Transposer<Device::CUDA, CI32, ArrayDimensionOrder::AFTP, ArrayDimensionOrder::ATPF>;
// template class BLADE_API Transposer<Device::CUDA, CI64, ArrayDimensionOrder::AFTP, ArrayDimensionOrder::ATPF>;
// template class BLADE_API Transposer<Device::CUDA, CU8,  ArrayDimensionOrder::AFTP, ArrayDimensionOrder::ATPF>;
// template class BLADE_API Transposer<Device::CUDA, CU16, ArrayDimensionOrder::AFTP, ArrayDimensionOrder::ATPF>;
// template class BLADE_API Transposer<Device::CUDA, CU32, ArrayDimensionOrder::AFTP, ArrayDimensionOrder::ATPF>;
// template class BLADE_API Transposer<Device::CUDA, CU64, ArrayDimensionOrder::AFTP, ArrayDimensionOrder::ATPF>;


}  // namespace Blade::Modules
