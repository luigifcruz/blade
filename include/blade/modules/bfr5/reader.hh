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

    const PhasorDimensions getDims() const {
        return {
            .B = this->bfr5.dim_info.nbeams,
            .A = this->bfr5.dim_info.nants,
            .F = this->bfr5.dim_info.nchan,
            .T = this->bfr5.dim_info.ntimes,
            .P = this->bfr5.dim_info.npol,
        };
    }

    const LLA getReferencePosition() const {
        return {
            .LON = calc_rad_from_degree(this->bfr5.tel_info.longitude),
            .LAT = calc_rad_from_degree(this->bfr5.tel_info.latitude),
            .ALT = this->bfr5.tel_info.altitude,
        };
    }

    constexpr const RA_DEC getPhaseCenterCoordinates() const {
        return {
            .RA = this->bfr5.obs_info.phase_center_ra,
            .DEC = this->bfr5.obs_info.phase_center_dec
        };
    }

    constexpr const std::vector<XYZ>& getAntennaPositions() const {
        return this->antennaPositions;
    }

    constexpr const std::vector<RA_DEC>& getBeamCoordinates() const {
        return this->beamCoordinates;
    }

    std::vector<F64> getBeamAntennaDelays() const {
        return std::vector<F64>(this->bfr5.delay_info.delays, this->bfr5.delay_info.delays + this->bfr5.delay_info.delay_elements);
    }

    std::vector<F64> getDelayTimes() const {
        return std::vector<F64>(this->bfr5.delay_info.time_array, this->bfr5.delay_info.time_array + this->bfr5.delay_info.time_array_elements);
    }

    const ArrayDimensions getAntennaCoefficientsDims() const {
        return {
            .A = this->bfr5.cal_info.cal_all_dims[2],
            .F = this->bfr5.cal_info.cal_all_dims[0],
            .T = 1,
            .P = this->bfr5.cal_info.cal_all_dims[1],
        };
    }

    std::vector<CF64> getAntennaCoefficients(const U64& numberOfFrequencyChannels = 0, const U64& frequencyChannelStartIndex = 0);

 private:
    // Variables

    Config config;
    const Input input;
    Output output;

    BFR5_file_t bfr5;

    // TODO: Update from vector to ArrayTensor. 
    std::vector<XYZ> antennaPositions;
    std::vector<RA_DEC> beamCoordinates;
};

}  // namespace Blade::Modules

#endif

