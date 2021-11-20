#ifndef BLADE_MODULES_BEAMFORMER_GENERIC_HH
#define BLADE_MODULES_BEAMFORMER_GENERIC_HH

#include <string>

#include "blade/base.hh"
#include "blade/module.hh"

namespace Blade::Modules::Beamformer {

class BLADE_API Generic : public module {
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

    Result run(const std::span<CF32>& input,
               const std::span<CF32>& phasors,
                     std::span<CF32>& output,
                     cudaStream_t cudaStream = 0);

 protected:
    const Config config;
};

}  // namespace Blade::Modules::Beamformer

#endif
