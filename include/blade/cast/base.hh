#ifndef BLADE_CAST_GENERIC_H
#define BLADE_CAST_GENERIC_H

#include "blade/base.hh"
#include "blade/kernel.hh"

namespace Blade::Cast {

class BLADE_API Generic : public Kernel {
public:
    struct Config {
        std::size_t blockSize = 512;
    };

    Generic(const Config & config);
    ~Generic();

    constexpr Config getConfig() const {
        return config;
    }

    Result run(const std::span<std::complex<int8_t>> &input,
                     std::span<std::complex<float>> &output,
                     cudaStream_t cudaStream = 0);

    Result run(const std::span<std::complex<float>> &input,
                     std::span<std::complex<half>> &output,
                     cudaStream_t cudaStream = 0);

private:
    const Config config;
    dim3 block;
    jitify2::ProgramCache<> cache;

    template<typename OT, typename IT>
    Result run(IT input, OT output, std::size_t size,
            cudaStream_t cudaStream = 0);
};

} // namespace Blade::Cast

#endif
