#ifndef BLADE_MODULES_POLARIZER_GENERIC_HH
#define BLADE_MODULES_POLARIZER_GENERIC_HH

#include "blade/base.hh"
#include "blade/module.hh"

namespace Blade::Modules {

template<typename IT, typename OT>
class BLADE_API Polarizer : public Module {
 public:
    // Configuration

    enum class Mode : uint8_t {
        BYPASS,
        XY2LR,
    };

    struct Config {
        Mode mode = Mode::XY2LR; 
        U64 blockSize = 512;
    };

    constexpr const Config& getConfig() const {
        return this->config;
    }

    // Input

    struct Input {
        const ArrayTensor<Device::CUDA, IT>& buf;
    };

    constexpr const ArrayTensor<Device::CUDA, IT>& getInputBuffer() const {
        return this->input.buf;
    }

    // Output 

    struct Output {
        ArrayTensor<Device::CUDA, OT> buf;
    };

    constexpr const ArrayTensor<Device::CUDA, OT>& getOutputBuffer() const {
        return this->output.buf;
    }

    // Taint Registers

    constexpr Taint getTaint() const {
        return Taint::MODIFIER;
    }

    // Constructor & Processing

    explicit Polarizer(const Config& config, const Input& input,
                       const cudaStream_t& stream);
    Result process(const cudaStream_t& stream, const U64& currentStepCount) final;

 private:
    // Variables

    const Config config;
    const Input input;
    Output output;

    // Expected Shape

    const ArrayShape getOutputBufferShape() const {
        return getInputBuffer().shape();
    }
};

}  // namespace Blade::Modules

#endif
