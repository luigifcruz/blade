#ifndef BLADE_CHANNELIZER_H
#define BLADE_CHANNELIZER_H

#include "blade/base.hh"
#include "blade/kernel.hh"

namespace Blade::Channelizer {

class BLADE_API Generic : public Kernel {
public:
    struct Config : public ArrayDims {
        std::size_t fftSize = 4;
        std::size_t blockSize = 1024;
    };

    Generic(const Config & config);
    ~Generic();

    constexpr Config getConfig() const {
        return config;
    }

    constexpr ArrayDims getOutputDims() const {
        auto cfg = config;
        cfg.NCHANS *= config.fftSize;
        cfg.NTIME /= config.fftSize;
        return cfg;
    }

    constexpr std::size_t getBufferSize() const {
        return config.NPOLS * config.NTIME * config.NANTS * config.NCHANS;
    }

    Result run(const std::span<std::complex<int8_t>> &input,
                     std::span<std::complex<int8_t>> &output);

private:
    const Config config;
    dim3 grid, block;
    std::string kernel;
    jitify2::ProgramCache<> cache;
};

} // namespace Blade::Channelizer

#endif
