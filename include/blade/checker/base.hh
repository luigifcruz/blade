#ifndef BLADE_CHECKER_GENERIC_H
#define BLADE_CHECKER_GENERIC_H

#include "blade/base.hh"
#include "blade/kernel.hh"

namespace Blade::Checker {

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

    template<typename IT, typename OT>
    Result run(const std::span<IT> &a,
               const std::span<OT> &b,
                     std::span<std::size_t> &result,
                     cudaStream_t cudaStream = 0);

    template<typename IT, typename OT>
    Result run(const std::span<std::complex<IT>> &a,
               const std::span<std::complex<OT>> &b,
                     std::span<std::size_t> &result,
                     cudaStream_t cudaStream = 0);

private:
    const Config config;
    dim3 block;
    jitify2::ProgramCache<> cache;

    template<typename IT, typename OT>
    Result run(IT a, OT b, std::span<std::size_t> &result, std::size_t size,
            std::size_t scale = 1, cudaStream_t cudaStream = 0);
};

} // namespace Blade::Checker

#endif
