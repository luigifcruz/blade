#ifndef BLADE_MODULES_CHANNELIZER_BASE_HH
#define BLADE_MODULES_CHANNELIZER_BASE_HH

#include <string>
#include <cufft.h>

#include "blade/base.hh"
#include "blade/module.hh"

#include "blade/modules/channelizer/callback.hh"

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
               MemoryTaint::MODIFIER;
    }

    // Constructor & Processing

    explicit Channelizer(const Config& config, const Input& input, 
                         const cudaStream_t& stream);
    Result process(const cudaStream_t& stream, const U64& currentComputeStep) final;
    ~Channelizer();

 private:
    // Variables 

    const Config config;
    const Input input;
    Output output;

    std::string post_kernel;
    dim3 post_grid, post_block;

    cufftHandle plan;
    std::string kernel_key;
    std::unique_ptr<Internal::Callback> callback;

    // Expected Shape

    const ArrayShape getOutputBufferShape() const {
        return ArrayShape({
            getInputBuffer().shape().numberOfAspects(),
            getInputBuffer().shape().numberOfFrequencyChannels() * config.rate,
            getInputBuffer().shape().numberOfTimeSamples() / config.rate,
            getInputBuffer().shape().numberOfPolarizations(),
        });
    }
};

}  // namespace Blade::Modules

#endif
