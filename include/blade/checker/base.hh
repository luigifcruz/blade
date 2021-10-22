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

    unsigned long long int run(const std::span<float> &a,
                               const std::span<float> &b);

    unsigned long long int run(const std::span<int8_t> &a,
                               const std::span<int8_t> &b);

    unsigned long long int run(const std::span<std::complex<float>> &a,
                               const std::span<std::complex<float>> &b);

    unsigned long long int run(const std::span<std::complex<int8_t>> &a,
                               const std::span<std::complex<int8_t>> &b);

private:
    const Config config;
    dim3 block;
    unsigned long long int* counter;
    jitify2::ProgramCache<> cache;

    template<typename IT, typename OT>
    unsigned long long int generic_run(IT a, OT b, std::size_t size,
            std::size_t scale = 1);
};

} // namespace Blade::Checker

#endif
