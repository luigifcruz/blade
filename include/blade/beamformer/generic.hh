#ifndef BLADE_BEAMFORMER_GENERIC_H
#define BLADE_BEAMFORMER_GENERIC_H

#include "blade/base.hh"
#include "blade/kernel.hh"

namespace Blade::Beamformer {

class BLADE_API Generic : public Kernel {
public:
    struct Config : ArrayDims {
        std::size_t blockSize;
    };

    Generic(const Config & config);
    virtual ~Generic() = default;

    constexpr Config getConfig() const {
        return config;
    }

    virtual constexpr std::size_t getInputSize() const = 0;
    virtual constexpr std::size_t getOutputSize() const = 0;
    virtual constexpr std::size_t getPhasorsSize() const = 0;

    Result run(const std::complex<int8_t>* input, const std::complex<float>* phasors, std::complex<float>* output);

protected:
    const Config config;
    std::string kernel;
    dim3 grid, block;
    jitify2::ProgramCache<> cache;
};

} // namespace Blade::Beamformer

#endif
