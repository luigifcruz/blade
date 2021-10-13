#ifndef BLADE_CHECKER_H
#define BLADE_CHECKER_H

#include "blade/base.hh"
#include "blade/kernel.hh"

namespace Blade {

class BLADE_API Checker : public Kernel {
public:
    struct Config {
        std::size_t inputSize;
        std::size_t blockSize = 256;
    };

    Checker(const Config & config);
    ~Checker();

    constexpr Config getConfig() const {
        return config;
    }

    constexpr std::size_t getInputSize() const {
        return config.inputSize;
    }

    unsigned long long int run(const std::complex<float>* input, const std::complex<float>* output);
    unsigned long long int run(const float* input, const float* output);
    unsigned long long int run(const int8_t* input, const int8_t* output);

private:
    const Config config;
    dim3 grid;
    dim3 block;
    unsigned long long int* counter;
    jitify2::ProgramCache<> cache;
};

} // namespace Blade

#endif
