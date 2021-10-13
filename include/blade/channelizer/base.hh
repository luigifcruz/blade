#ifndef BLADE_CHANNELIZER_H
#define BLADE_CHANNELIZER_H

#include "blade/base.hh"
#include "blade/kernel.hh"

namespace Blade {

class BLADE_API Channelizer : public Kernel {
public:
    struct Config : public ArrayDims {
        std::size_t fftSize = 4;
        std::size_t blockSize = 256;
    };

    Channelizer(const Config & config);
    ~Channelizer();

    constexpr Config getConfig() const {
        return config;
    }

    constexpr ArrayDims getOutputDims() const {
        auto cfg = config;
        cfg.NCHANS *= config.fftSize;
        cfg.NTIME /= config.fftSize;
        return cfg;
    }

    // In this case, the input and output will be the same.
    constexpr std::size_t getInputSize() const {
        return config.NPOLS * config.NTIME * config.NANTS * config.NCHANS;
    }

    Result run(const std::complex<int8_t>* input);

private:
    const Config config;
    dim3 grid, block;
    std::string kernel;
    jitify2::ProgramCache<> cache;
};

} // namespace Blade

#endif
