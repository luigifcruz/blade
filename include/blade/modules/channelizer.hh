#ifndef BLADE_MODULES_CHANNELIZER_HH
#define BLADE_MODULES_CHANNELIZER_HH

#include <string>
#include <cufft.h>

#include "blade/base.hh"
#include "blade/module.hh"

namespace Blade::Modules {

template<typename IT, typename OT>
class BLADE_API Channelizer : public Module {
 public:
    // Configuration

    struct Config {
        U64 rate = 4;

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

    constexpr const MemoryTaint getMemoryTaint() {
        return MemoryTaint::CONSUMER |
               MemoryTaint::PRODUCER;
    }

    // Constructor & Processing

    explicit Channelizer(const Config& config, const Input& input);
    const Result process(const cudaStream_t& stream) final;
    ~Channelizer();

 private:
    // Variables 

    const Config config;
    const Input input;
    Output output;

    std::string pre_kernel;
    dim3 pre_grid, pre_block;

    std::string post_kernel;
    dim3 post_grid, post_block;

    ArrayTensor<Device::CUDA, OT> buffer;
    ArrayTensor<Device::CPU | Device::CUDA, U64> indices;

    cufftHandle plan;
    std::string kernel_key;

    // Expected Dimensions

    const ArrayDimensions getOutputBufferDims() const {
        return {
            .A = getInputBuffer().dims().numberOfAspects(),
            .F = getInputBuffer().dims().numberOfFrequencyChannels() * config.rate,
            .T = getInputBuffer().dims().numberOfTimeSamples() / config.rate,
            .P = getInputBuffer().dims().numberOfPolarizations(),
        };
    }
};

}  // namespace Blade::Modules

#endif
