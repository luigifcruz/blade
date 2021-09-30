#ifndef BL_CHECKER_H
#define BL_CHECKER_H

#include "bl-beamformer/type.hh"
#include "bl-beamformer/helpers.hh"

namespace BL {

class Checker {
public:
    struct Config {
        size_t len;
        size_t block = 256;
    };

    Checker(const Config & config);
    ~Checker();

    constexpr size_t inputLen() const {
        return config.len;
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

} // namespace BL::Checker

#endif
