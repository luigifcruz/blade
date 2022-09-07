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
    struct Config {
        U64 rate = 4;

        U64 blockSize = 512;
    };

    struct Input {
        const ArrayTensor<Device::CUDA, IT>& buf;
    };

    struct Output {
        ArrayTensor<Device::CUDA, OT> buf;
    };

    explicit Channelizer(const Config& config, const Input& input);

    constexpr const ArrayTensor<Device::CUDA, IT>& getInput() const {
        return this->input.buf;
    }

    constexpr const ArrayTensor<Device::CUDA, OT>& getOutput() const {
        return this->output.buf;
    }

    constexpr const Config& getConfig() const {
        return this->config;
    }

    const Result process(const cudaStream_t& stream = 0) final;

 private:
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

    constexpr const ArrayTensorDimensions getOutputDims() const {
        return {
            getInput().numberOfAspects(),
            getInput().numberOfFrequencyChannels() * config.rate,
            getInput().numberOfTimeSamples() / config.rate,
            getInput().numberOfPolarizations(),
        };
    }
};

}  // namespace Blade::Modules

#endif
