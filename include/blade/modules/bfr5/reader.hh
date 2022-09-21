#ifndef BLADE_MODULES_BFR5_READER_HH
#define BLADE_MODULES_BFR5_READER_HH

#include <filesystem>
#include <string>

#include "blade/base.hh"
#include "blade/module.hh"

extern "C" {
#include "bfr5.h"
#include "radiointerferometryc99.h"
}

namespace Blade::Modules::Bfr5 {

class BLADE_API Reader : public Module {
 public:
    // Configuration

    struct Config {
        std::string filepath;

        U64 blockSize = 512;
    };

    // Input

    struct Input {
    };

    // Output

    struct Output {
    };

    // Constructor & Processing

    explicit Reader(const Config& config, const Input& input);

    // Miscellaneous

    const PhasorTensorDimensions getTotalDims() const {
        return {
            .B = this->bfr5.dim_info.nbeams,
            .A = this->bfr5.dim_info.nants,
            .F = this->bfr5.dim_info.nchan,
            .T = this->bfr5.dim_info.ntimes,
            .P = this->bfr5.dim_info.npol,
        };
    }

    constexpr const U64 getTotalNumberOfAntennas() const {
        return this->bfr5.dim_info.nants;
    }

    constexpr const U64 getTotalNumberOfBeams() const {
        return this->bfr5.dim_info.nbeams;
    }

    constexpr const U64 getTotalNumberOfFrequencyChannels() const {
        return this->bfr5.dim_info.nchan;
    }

    constexpr const U64 getTotalNumberOfPolarizations() const {
        return this->bfr5.dim_info.npol;
    }

    constexpr const LLA getReferencePosition() const {
        return {
            this->bfr5.tel_info.latitude,
            this->bfr5.tel_info.longitude,
            this->bfr5.tel_info.altitude
        };
    }

    constexpr const RA_DEC getBoresightCoordinate() const {
        return {
            this->bfr5.obs_info.phase_center_ra,
            this->bfr5.obs_info.phase_center_dec
        };
    }

    const std::vector<XYZ> getAntennaPositions() const {
        return this->antennaPositions;
    }

    const std::vector<RA_DEC> getBeamCoordinates() const {
        return this->beamCoordinates;
    }

    void fillAntennaCalibrations(const U64& numberOfFrequencyChannels,
                                const U64& channelizerRate, 
                                ArrayCoefficientTensor<Device::CPU, CF64>& antennaCalibrations);

 private:
    // Variables

    Config config;
    const Input input;
    Output output;

    BFR5_file_t bfr5;

    std::vector<XYZ> antennaPositions;
    std::vector<RA_DEC> beamCoordinates;
};

}  // namespace Blade::Modules

#endif

