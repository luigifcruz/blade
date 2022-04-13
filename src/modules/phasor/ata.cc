#include "blade/modules/phasor/ata.hh"

extern "C" {
#include "radiointerferometryc99.h"
}

namespace Blade::Modules::Phasor {

template<typename OT>
ATA<OT>::ATA(const typename Generic<OT>::Config& config,
             const typename Generic<OT>::Input& input)
        : Generic<OT>(config, input) {
    //  Preallocating arrays.
    antennas_xyz.resize(this->config.numberOfAntennas);
    boresight_uvw.resize(this->config.numberOfAntennas);
    source_uvw.resize(this->config.numberOfAntennas);
    boresightDelay.resize(this->config.numberOfAntennas);
    BL_CHECK_THROW(this->InitInput(relativeDelay, 
        this->config.numberOfAntennas * this->config.numberOfBeams));

    //  Copy Earth Centered XYZ Antenna Coordinates (XYZ) to Receiver (UVW).
    antennas_xyz = this->config.antennaPositions;

    //  Translate Antenna Position (ECEF) to Reference Position (XYZ).
    calc_position_to_xyz_frame_from_ecef(
        (F64*)antennas_xyz.data(),
        this->config.numberOfAntennas,
        this->config.arrayReferencePosition.LON,
        this->config.arrayReferencePosition.LAT,
        this->config.arrayReferencePosition.ALT);

    BL_CHECK_THROW(this->InitOutput(this->output.phasors, getPhasorsSize()));
}

template<typename OT>
Result ATA<OT>::preprocess(const cudaStream_t& stream) {
    HA_DEC boresight_ha_dec = {0.0, 0.0};

    //  Convert Boresight RA & Declination to Hour Angle & Declination.
    calc_ha_dec_rad(
        this->config.boresightCoordinate.RA,
        this->config.boresightCoordinate.DEC,
        this->config.arrayReferencePosition.LON,
        this->config.arrayReferencePosition.LAT,
        this->config.arrayReferencePosition.ALT,
        this->input.frameJulianDate,
        this->input.differenceUniversalTime1,
        &boresight_ha_dec.HA,
        &boresight_ha_dec.DEC);

    //  Copy Reference Position (XYZ) to Boresight Position (UVW).
    for (U64 i = 0; i < antennas_xyz.size(); i++) {
        boresight_uvw[i] = reinterpret_cast<const UVW&>(antennas_xyz[i]);
    }

    //  Rotate Reference Position (UVW) towards Boresight (HA, Dec).
    calc_position_to_uvw_frame_from_xyz(
        (F64*)boresight_uvw.data(),
        this->config.numberOfAntennas,
        boresight_ha_dec.HA,
        boresight_ha_dec.DEC,
        this->config.arrayReferencePosition.LON);

    //  Calculate delay for boresight (Ti = (Wi - Wr) / C).
    for (U64 i = 0; i < this->config.numberOfAntennas; i++) {
        boresightDelay[i] = (
            boresight_uvw[i].W - 
            boresight_uvw[this->config.referenceAntennaIndex].W
        ) / BL_PHYSICAL_CONSTANT_C; 
    }

    eraASTROM astrom;

    // Convert source RA & Decligation to Hour Angle (Part A).
    calc_ha_dec_rad_a(
        this->config.arrayReferencePosition.LON,
        this->config.arrayReferencePosition.LAT, 
        this->config.arrayReferencePosition.ALT,
        this->input.frameJulianDate,
        this->input.differenceUniversalTime1,
        &astrom);

    for (U64 b = 0; b < this->config.numberOfBeams; b++) {
        //  Copy Reference Position (XYZ) to Source Position (UVW).
        for (U64 i = 0; i < antennas_xyz.size(); i++) {
            source_uvw[i] = reinterpret_cast<const UVW&>(antennas_xyz[i]);
        }

        HA_DEC source_ha_dec = {0.0, 0.0};

        //  Convert source RA & Decligation to Hour Angle (Part B).
        calc_ha_dec_rad_b(
            this->config.beamCoordinates[b].RA,
            this->config.beamCoordinates[b].DEC,
            &astrom, 
            &source_ha_dec.HA,
            &source_ha_dec.DEC);

        //  Rotate Reference Position (UVW) towards Source (HA, Dec).
        calc_position_to_uvw_frame_from_xyz(
            (F64*)source_uvw.data(),
            this->config.numberOfAntennas,
            boresight_ha_dec.HA,
            boresight_ha_dec.DEC,
            this->config.arrayReferencePosition.LON);

        for (size_t i = 0; i < this->config.numberOfAntennas; i++) {
            printf("+ %zu: U: %lf V: %lf W: %lf\n", i, source_uvw[i].U, source_uvw[i].V, source_uvw[i].W);
        }

        //  Calculate delay for off-center source and subtract 
        //  from boresight (TPi = Ti - ((WPi - WPr) / C)).
        for (U64 i = 0; i < this->config.numberOfAntennas; i++) {
            relativeDelay[(b * this->config.numberOfAntennas) + i] = 
                boresightDelay[i] - (
                    (
                        source_uvw[i].W -
                        source_uvw[this->config.referenceAntennaIndex].W
                    ) / BL_PHYSICAL_CONSTANT_C
                );
        }

        for (U64 i = 0; i < this->config.numberOfAntennas; i++) {
            printf("%.13lf\n", relativeDelay[i]);
        }
    }

    //  TODO: Add hint for CUDA Unified Memory.

    return Result::SUCCESS;
}

template<typename OT>
Result ATA<OT>::process(const cudaStream_t& stream) {
    //  TODO: Add phasors kernel.

    return Result::SUCCESS;
}

template class BLADE_API ATA<CF32>;
template class BLADE_API ATA<CF64>;

}  // namespace Blade::Modules::Phasor