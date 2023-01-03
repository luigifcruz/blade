#ifndef BLADE_MODULES_TRANSPOSER_GENERIC_HH
#define BLADE_MODULES_TRANSPOSER_GENERIC_HH

#include "blade/base.hh"
#include "blade/module.hh"

namespace Blade {

enum class BLADE_API ArrayDimensionOrder : uint8_t {
    AFTP    = 0,
    ATPF    = 1,
};

constexpr const char* ArrayDimensionOrderName(const ArrayDimensionOrder order) {
    switch (order) {
        case ArrayDimensionOrder::AFTP:
            return "AFTP";
        case ArrayDimensionOrder::ATPF:
            return "ATPF";
        default:
            return "????";
    }
}

} // namespace Blade

namespace Blade::Modules {

template<Device Dev, typename ElementType, ArrayDimensionOrder InputOrder, ArrayDimensionOrder OutputOrder>
class BLADE_API Transposer : public Module {
    public:
    // Configuration

    struct Config {
        U64 blockSize = 512;
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        const ArrayTensor<Dev, ElementType>& buf;
    };
    
    constexpr const ArrayTensor<Dev, ElementType>& getInputBuffer() const {
        return this->input.buf;
    }

    // Output

    struct Output {
        ArrayTensor<Dev, ElementType> buf;
    };

    constexpr const ArrayTensor<Dev, ElementType>& getOutputBuffer() const {
        if (this->bypass) {
            return this->input.buf;
        } else {
            return this->output.buf;
        }
    }

    // Constructor & Processing

    explicit Transposer(const Config& config, const Input& input);
    const Result process(const cudaStream_t& stream = 0) final;

    private:
    // Variables

    const Config config;
    const Input input;
    BOOL bypass;
    Output output;
};

} // namespace Blade::Modules::Transposer

#endif
