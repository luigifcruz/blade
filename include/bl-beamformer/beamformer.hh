#ifndef BL_BEAMFORMER_H
#define BL_BEAMFORMER_H

#include "bl-beamformer/type.hh"
#include "bl-beamformer/helpers.hh"

namespace BL {

class BL_API Beamformer {
public:
    enum Kernel {
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
        Kernel kernel;
    };

    Beamformer(const Config & config);
    ~Beamformer();

    constexpr std::size_t inputLen() const {
        return config.NANTS*config.NCHANS*config.NTIME*config.NPOLS;
    };

    constexpr std::size_t outputLen() const {
        return config.NBEAMS*config.NTIME*config.NCHANS*config.NPOLS;
    };

    constexpr std::size_t phasorLen() const {
        switch (config.kernel) {
            case Kernel::ATA:
            case Kernel::ATA_4P_FFT:
                return config.NBEAMS*config.NANTS*config.NCHANS*config.NPOLS;
            case Kernel::MEERKAT:
                return config.NBEAMS*config.NANTS;
        }
        return 0;
    };

    Result run(const std::complex<int8_t>* input, const std::complex<float>* phasor, std::complex<float>* output);

private:
    const Config config;
    std::string kernel;
    dim3 grid;
    dim3 block;
    jitify2::ProgramCache<> cache;
};

} // namespace BL

#endif
