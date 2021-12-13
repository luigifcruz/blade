#ifndef BLADE_MODULES_CHANNELIZER_HH
#define BLADE_MODULES_CHANNELIZER_HH

#include <string>

#include "blade/base.hh"
#include "blade/module.hh"

namespace Blade::Modules {

template<typename IT, typename OT>
class BLADE_API Channelizer : public Module {
 public:
    struct Config {
        ArrayDims dims;
        std::size_t fftSize = 4;
        std::size_t blockSize = 512;
    };

    struct Input {
        Vector<Device::CUDA, IT>& buf;
    };

    struct Output {
        Vector<Device::CUDA, OT> buf;
    };

    explicit Channelizer(const Config& config, const Input& input);

    constexpr Vector<Device::CUDA, IT>& getInput() {
        return this->input.buf;
    }

    constexpr const Vector<Device::CUDA, OT>& getOutput() const {
        return this->output.buf;
    }

    constexpr const Config& getConfig() const {
        return this->config;
    }

    constexpr const ArrayDims getOutputDims() {
        auto cfg = config.dims;
        cfg.NCHANS *= config.fftSize;
        cfg.NTIME /= config.fftSize;
        return cfg;
    }

    constexpr const std::size_t getBufferSize() {
        return config.dims.NPOLS * config.dims.NTIME *
            config.dims.NANTS * config.dims.NCHANS;
    }

    Result process(const cudaStream_t& stream = 0) final;

 private:
    const Config config;
    const Input input;
    Output output;
};

}  // namespace Blade::Modules

#endif
