#ifndef BLADE_MODULES_KURTOSIS_GENERIC_HH
#define BLADE_MODULES_KURTOSIS_GENERIC_HH

#include "blade/base.hh"
#include "blade/module.hh"

namespace Blade::Modules {

template<typename IT, typename OT>
class BLADE_API Kurtosis : public Module {
 public:
    // Configuration

    struct Config {
        bool debugMode = false;
        int kurtosisChannelLength = 256;
        int numberOfKurtosisStddev = 5;
        int numberOfMaskRuns = 128;
        std::string maskFilePath = "./blade_out.bin";
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
        ArrayTensor<Device::CUDA, U8> mask;
    };

    constexpr const ArrayTensor<Device::CUDA, OT>& getOutputBuffer() const {
        return this->output.buf;
    }

    constexpr const ArrayTensor<Device::CUDA, U8>& getOutputMask() const {
        return this->output.mask;
    }

    // Taint Registers

    constexpr Taint getTaint() const {
        return Taint::MODIFIER;
    }

    std::string name() const {
        return "Kurtosis";
    }

    // Constructor & Processing

    explicit Kurtosis(const Config& config, const Input& input, const Stream& stream = {});
    ~Kurtosis();
    Result process(const U64& currentStepCount, const Stream& stream = {}) final;

 private:
    // Variables

    const Config config;
    const Input input;
    Output output;
    int maskCounter;
    std::ofstream maskOutFile;

    // Expected Shape

    const ArrayShape getOutputBufferShape() const {
        return getInputBuffer().shape();
    }

    Result writeMaskToDisk();
};

}  // namespace Blade::Modules

#endif
