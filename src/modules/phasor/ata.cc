#include "blade/modules/phasor/ata.hh"

extern "C" {
#include "radiointerferometryc99.h"
}

// [Documentation - ATA Delays Processor] 
//
// [Legend]:
//      - A: Number of Antennas. 
//      - B: Number of Beams.
//      - N: Number of Blocks.
//
// [Pipeline]:
// 1. Start with Earth Centered Antenna Positions (ECEF).
// 2. Translate Earth Centered Antenna Positions (ECEF) to Array Centered Antenna Positions (XYZ).
//      - Runs on initialization for each antenna (A).
//      - Based on "calc_position_to_xyz_frame_from_ecef" method.
//      - Depends on the Array Center Reference Longitude, Latitude, and Altitude values.
// 3. Rotate Array Centered Antenna Position (XYZ) towards Boresight (UVW).
//      - Runs on each block for each antenna (A*N). 
//      - Based on "calc_position_to_uvw_frame_from_xyz" method.
//      - Depends on the Hour Angle & Declination values of the Boresight. 
// 4. Calculate time delay on Boresight.
//      - Runs on each block for each antenna (A*N). 
//      - Defined by Ti = (Wi - Wr) / C.
//          - Ti = Time Delay (s) of the signal from Reference Antenna.
//          - Wi = Distance (m) of the current antenna to the boresight.
//          - Wr = Distance (m) of the reference antenna to the boresight.
//          - C  = Speed of Light (m/s).
//      - Depends on the Array Centered Antenna Position (XYZ) and Hour Angle & Declination of the Boresight.
// 5. Generate Hour Angle & Declination from RA & Declination according to time.
//      - Part A runs on each block (N), and Part B runs on each block for every beam (B*N).
//      - Based on "calc_ha_dec_rad_a" (Part A) and "calc_ha_dec_rad_b" (Part B) methods. 
//      - Depends on the RA & Declination values of the Source.  
// 6. Rotate Array Centered Antenna Position (XYZ) towards Source (UVW).
//      - Runs on each block for each antenna for every beam (A*B*N). 
//      - Based on "calc_position_to_uvw_frame_from_xyz" method.
//      - Depends on the Hour Angle & Declination values of the Source. 
// 7. Calculate time delay on Source.
//      - Runs on each block for each antenna for every beam (A*B*N).
//      - Defined by TPi = Ti - ((WPi - WPr) / C).
//          - TPi = Time Delay (s) from Boresight to Source.
//          - Ti = Time Delay (s) of the signal from Reference Antenna.
//          - WPi = Distance (m) of the current antenna to the signal source.
//          - WPr = Distance (m) of the reference antenna to the signal source.
//          - C  = Speed of Light (m/s).

namespace Blade::Modules::Phasor {

template<typename OT>
ATA<OT>::ATA(const typename Generic<OT>::Config& config,
             const typename Generic<OT>::Input& input)
        : Generic<OT>(config, input) {
    //  Resizing array to the required length.
    antennasXyz.resize(this->config.numberOfAntennas);
    boresightUvw.resize(this->config.numberOfAntennas);
    sourceUvw.resize(this->config.numberOfAntennas);
    boresightDelay.resize(this->config.numberOfAntennas);

    //  Copy Earth Centered XYZ Antenna Coordinates (XYZ) to Receiver (UVW).
    antennasXyz = this->config.antennaPositions;

    //  Translate Antenna Position (ECEF) to Reference Position (XYZ).
    calc_position_to_xyz_frame_from_ecef(
        (F64*)antennasXyz.data(),
        this->config.numberOfAntennas,
        this->config.arrayReferencePosition.LON,
        this->config.arrayReferencePosition.LAT,
        this->config.arrayReferencePosition.ALT);

    BL_CHECK_THROW(this->InitOutput(this->output.delays, getDelaysSize()));
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
        this->input.frameJulianDate[0],
        this->input.frameDut1[0],
        &boresight_ha_dec.HA,
        &boresight_ha_dec.DEC);

    //  Copy Reference Position (XYZ) to Boresight Position (UVW).
    for (U64 i = 0; i < antennasXyz.size(); i++) {
        boresightUvw[i] = reinterpret_cast<const UVW&>(antennasXyz[i]);
    }

    //  Rotate Reference Position (UVW) towards Boresight (HA, Dec).
    calc_position_to_uvw_frame_from_xyz(
        (F64*)boresightUvw.data(),
        this->config.numberOfAntennas,
        boresight_ha_dec.HA,
        boresight_ha_dec.DEC,
        this->config.arrayReferencePosition.LON);

    //  Calculate delay for boresight (Ti = (Wi - Wr) / C).
    for (U64 a = 0; a < this->config.numberOfAntennas; a++) {
        boresightDelay[a] = (
            boresightUvw[a].W - 
            boresightUvw[this->config.referenceAntennaIndex].W
        ) / BL_PHYSICAL_CONSTANT_C; 
    }

    eraASTROM astrom;

    // Convert source RA & Declination to Hour Angle (Part A).
    calc_ha_dec_rad_a(
        this->config.arrayReferencePosition.LON,
        this->config.arrayReferencePosition.LAT, 
        this->config.arrayReferencePosition.ALT,
        this->input.frameJulianDate[0],
        this->input.frameDut1[0],
        &astrom);

    for (U64 b = 0; b < this->config.numberOfBeams; b++) {
        //  Copy Reference Position (XYZ) to Source Position (UVW).
        for (U64 i = 0; i < antennasXyz.size(); i++) {
            sourceUvw[i] = reinterpret_cast<const UVW&>(antennasXyz[i]);
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
            (F64*)sourceUvw.data(),
            this->config.numberOfAntennas,
            source_ha_dec.HA,
            source_ha_dec.DEC,
            this->config.arrayReferencePosition.LON);

        //  Calculate delay for off-center source and subtract 
        //  from boresight (TPi = Ti - ((WPi - WPr) / C)).
        for (U64 a = 0; a < this->config.numberOfAntennas; a++) {
            this->output.delays[(b * this->config.numberOfAntennas) + a] = 
                (
                    (
                        sourceUvw[a].W -
                        sourceUvw[this->config.referenceAntennaIndex].W
                    ) / BL_PHYSICAL_CONSTANT_C
                ) - boresightDelay[a];
        }
    }

    for (U64 b = 0; b < this->config.numberOfBeams; b++) {
        const U64 beamOffset = (b * 
                                this->config.numberOfAntennas * 
                                this->config.numberOfFrequencyChannels * 
                                this->config.numberOfPolarizations); 

        for (U64 a = 0; a < this->config.numberOfAntennas; a++) {
            const U64 antennaOffset = (a *
                                       this->config.numberOfFrequencyChannels *
                                       this->config.numberOfPolarizations);

            const F64 delay = this->output.delays[(b * this->config.numberOfAntennas) + a];
            const F64 fringe = this->config.rfFrequencyHz - (this->config.totalBandwidthHz / 2.0);
            const CF64 fringeRateExp(0, -2 * BL_PHYSICAL_CONSTANT_PI * delay * fringe); 

            for (U64 f = 0; f < this->config.numberOfFrequencyChannels; f++) {
                const U64 frequencyOffset = (f * this->config.numberOfPolarizations);

                const F64 freq = (f + this->config.frequencyStartIndex) * this->config.channelBandwidthHz;
                const CF64 phasorsExp(0, -2 * BL_PHYSICAL_CONSTANT_PI * delay * freq); 
                const CF64 phasor = std::exp(phasorsExp + fringeRateExp);

                for (U64 p = 0; p < this->config.numberOfPolarizations; p++) {
                    const U64 polarizationOffset = p;

                    const U64 calibrationIndex = antennaOffset + frequencyOffset + polarizationOffset; 
                    const U64 phasorsIndex = beamOffset + calibrationIndex;

                    this->output.phasors[phasorsIndex] = phasor; // * this->config.antennaCalibrations[calibrationIndex];
                }
            }
        }
    }

    return Result::SUCCESS;
}

template class BLADE_API ATA<CF32>;
template class BLADE_API ATA<CF64>;

}  // namespace Blade::Modules::Phasor
