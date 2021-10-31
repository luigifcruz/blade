#ifndef BLADE_BEAMFORMER_GENERIC_H
#define BLADE_BEAMFORMER_GENERIC_H

#include <string>

#include "blade/base.hh"
#include "blade/kernel.hh"

namespace Blade::Beamformer {

class BLADE_API Generic : public Kernel {
 public:
    class Test;

    struct Config {
        ArrayDims dims;
        std::size_t blockSize = 512;
    };

    explicit Generic(const Config& config);
    virtual ~Generic() = default;

    constexpr Config getConfig() const {
        return config;
    }

    virtual constexpr std::size_t getInputSize() const = 0;
    virtual constexpr std::size_t getOutputSize() const = 0;
    virtual constexpr std::size_t getPhasorsSize() const = 0;

    Result run(const std::span<std::complex<F32>>& input,
               const std::span<std::complex<F32>>& phasors,
                     std::span<std::complex<F32>>& output,
                     cudaStream_t cudaStream = 0);

 protected:
    const Config config;
    std::string kernel;
    dim3 grid, block;
    jitify2::ProgramCache<> cache;
};

}  // namespace Blade::Beamformer

#endif  // BLADE_INCLUDE_BLADE_BEAMFORMER_GENERIC_HH_
