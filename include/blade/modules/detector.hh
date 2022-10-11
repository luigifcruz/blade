#ifndef BLADE_MODULES_DETECTOR_HH
#define BLADE_MODULES_DETECTOR_HH

#include <string>

#include "blade/base.hh"
#include "blade/module.hh"

namespace Blade::Modules {

template<typename IT, typename OT>
class BLADE_API Detector : public Module {
 public:
    // Configuration

    struct Config {
        U64 integrationSize;
        U64 numberOfOutputPolarizations = 4;

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

    // Constructor & Processing

    explicit Detector(const Config& config, const Input& input);
    const Result process(const cudaStream_t& stream = 0) final;

 private:
    // Variables 

    const Config config;
    const Input input;
    Output output;

    // Expected Dimensions

    const ArrayDimensions getOutputBufferDims() const {
        return {
            .A = getInputBuffer().dims().numberOfAspects(),
            .F = getInputBuffer().dims().numberOfFrequencyChannels(),
            .T = getInputBuffer().dims().numberOfTimeSamples() / config.integrationSize,
            .P = config.numberOfOutputPolarizations,
        }; 
    }
};

}  // namespace Blade::Modules

#endif

