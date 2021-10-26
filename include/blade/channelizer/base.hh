#ifndef BLADE_CHANNELIZER_H
#define BLADE_CHANNELIZER_H

#include <string>

#include "blade/base.hh"
#include "blade/kernel.hh"

namespace Blade {

class BLADE_API Channelizer : public Kernel {
 public:
    class Test;

    struct InternalConfig {
        std::size_t fftSize = 4;
        std::size_t blockSize = 512;
    };

    struct Config : ArrayDims, InternalConfig {};

    explicit Channelizer(const Config& config);
    ~Channelizer();

    constexpr Config getConfig() const {
        return config;
    }

    constexpr ArrayDims getOutputDims(std::size_t beams = 1) const {
        auto cfg = config;
        cfg.NBEAMS = beams;
        cfg.NCHANS *= config.fftSize;
        cfg.NTIME /= config.fftSize;
        return cfg;
    }

    constexpr std::size_t getBufferSize() const {
        return config.NPOLS * config.NTIME * config.NANTS * config.NCHANS;
    }

    Result run(const std::span<std::complex<float>>& input,
                     std::span<std::complex<float>>& output,
                     cudaStream_t cudaStream = 0);

 private:
    const Config config;
    dim3 grid, block;
    std::string kernel;
    jitify2::ProgramCache<> cache;
};

}  // namespace Blade

#endif  // BLADE_INCLUDE_BLADE_CHANNELIZER_BASE_HH_
