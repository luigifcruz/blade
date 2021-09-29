#ifndef BL_BEAMFORMER_H
#define BL_BEAMFORMER_H

#include "bl-beamformer/type.hh"
#include "bl-beamformer/helpers.hh"

namespace BL {

class Beamformer {
public:
    struct Config {
        size_t NBEAMS;
        size_t NANTS;
        size_t NCHANS;
        size_t NTIME;
        size_t NPOLS;
        size_t TBLOCK;
    };

    Beamformer(const Config & config);

    constexpr size_t inputLen() const {
        return config.NANTS*config.NCHANS*config.NTIME*config.NPOLS;
    };

    constexpr size_t phasorLen() const {
        return config.NBEAMS*config.NANTS*config.NCHANS*config.NPOLS;
    };

    constexpr size_t outputLen() const {
        return config.NBEAMS*config.NTIME*config.NCHANS*config.NPOLS;
    };

    Result run(const std::complex<int8_t>* input, const std::complex<float>* phasor, std::complex<float>* output);

private:
    const Config config;
    std::string kernel;
    dim3 grid;
    dim3 block;
};

} // namespace BL

#endif
