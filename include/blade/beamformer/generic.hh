#ifndef BLADE_BEAMFORMER_GENERIC_H
#define BLADE_BEAMFORMER_GENERIC_H

#include "blade/base.hh"
#include "blade/kernel.hh"

namespace Blade::Beamformer {

class BLADE_API Generic : public Kernel {
public:
    struct Config {
        std::size_t NBEAMS;
        std::size_t NANTS;
        std::size_t NCHANS;
        std::size_t NTIME;
        std::size_t NPOLS;
        std::size_t TBLOCK;
    };

    Generic(const Config & config);
    virtual ~Generic() = default;

    virtual constexpr std::size_t inputLen() const = 0;
    virtual constexpr std::size_t outputLen() const = 0;
    virtual constexpr std::size_t phasorsLen() const = 0;

    Result run(const std::complex<int8_t>* input, const std::complex<float>* phasors, std::complex<float>* output);

    constexpr Config getConfig() const {
        return config;
    }

protected:
    const Config config;
    std::string kernel;
    dim3 grid, block;
    jitify2::ProgramCache<> cache;
};

} // namespace Blade::Beamformer

#endif
