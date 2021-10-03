#ifndef BLADE_KERNELS_BEAMFORMER_H
#define BLADE_KERNELS_BEAMFORMER_H

#include "blade/kernels/base.hh"

namespace Blade::Kernel {

class BLADE_API Beamformer : public Generic {
public:
    enum Recipe {
        ATA,
        ATA_4P_FFT,
        MEERKAT,
    };

    struct Config {
        std::size_t NBEAMS;
        std::size_t NANTS;
        std::size_t NCHANS;
        std::size_t NTIME;
        std::size_t NPOLS;
        std::size_t TBLOCK;
        Recipe recipe;
    };

    Beamformer(const Config & config);
    ~Beamformer();

    constexpr std::size_t inputLen() const {
        return config.NANTS*config.NCHANS*config.NTIME*config.NPOLS;
    };

    constexpr std::size_t outputLen() const {
        return config.NBEAMS*config.NTIME*config.NCHANS*config.NPOLS;
    };

    constexpr std::size_t phasorsLen() const {
        switch (config.recipe) {
            case Recipe::ATA:
            case Recipe::ATA_4P_FFT:
                return config.NBEAMS*config.NANTS*config.NCHANS*config.NPOLS;
            case Recipe::MEERKAT:
                return config.NBEAMS*config.NANTS;
        }
        return 0;
    };

    Result run(const std::complex<int8_t>* input, const std::complex<float>* phasors, std::complex<float>* output);

private:
    const Config config;
    std::string kernel;
    dim3 grid, block;
    jitify2::ProgramCache<> cache;
};

} // namespace Blade::Kernel

#endif
