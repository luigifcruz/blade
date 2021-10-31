#ifndef BLADE_CHANNELIZER_H
#define BLADE_CHANNELIZER_H

#include <string>

#include "blade/base.hh"
#include "blade/kernel.hh"

namespace Blade {

class BLADE_API Channelizer : public Kernel {
 public:
    class Test;

    struct Config {
        ArrayDims dims;
        std::size_t fftSize = 4;
        std::size_t blockSize = 512;
    };

    explicit Channelizer(const Config& config);
    ~Channelizer();

    constexpr Config getConfig() const {
        return config;
    }

    constexpr ArrayDims getOutputDims() const {
        auto cfg = config.dims;
        cfg.NCHANS *= config.fftSize;
        cfg.NTIME /= config.fftSize;
        return cfg;
    }

    constexpr std::size_t getBufferSize() const {
        return config.dims.NPOLS * config.dims.NTIME *
            config.dims.NANTS * config.dims.NCHANS;
    }

    Result run(const std::span<CF32>& input,
                     std::span<CF32>& output,
                     cudaStream_t cudaStream = 0);

 private:
    const Config config;
    dim3 grid, block;
    std::string kernel;
    jitify2::ProgramCache<> cache;
};

}  // namespace Blade

#endif  // BLADE_INCLUDE_BLADE_CHANNELIZER_BASE_HH_
