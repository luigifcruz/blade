#include <jetstream/backend/devices/cpu/helpers.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

extern "C" {
#include "radiointerferometryc99.h"
}

#include "module_impl.hh"

namespace Jetstream::Modules {

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

struct Triplet {
    F64 a;
    F64 b;
    F64 c;
};

struct PhasorImplNativeCpu : public PhasorImpl,
                           public NativeCpuRuntimeContext,
                           public Scheduler::Context {
 public:
    Result create() final;

    Result computeSubmit() override;

 private:
    Result kernelCF32();

    std::function<Result()> kernel;

    // Tensor antennasXyz;
    std::vector<Triplet> antennasXyz;
    std::vector<Triplet> boresightUvw;
    std::vector<Triplet> sourceUvw;
    std::vector<F64> boresightDelay;
};

Result PhasorImplNativeCpu::create() {
    // Create parent.
    JST_CHECK(PhasorImpl::create());
    
    const auto& nofAntennas = antennaPositionTensor.shape()[0];
    // JST_CHECK(antennasXyz.create(
    //     DeviceType::CPU,
    //     DataType::F64,
    //     {nofAntennas, 3}
    // ));
    antennasXyz.resize(nofAntennas);
    boresightUvw.resize(nofAntennas);
    sourceUvw.resize(nofAntennas);
    boresightDelay.resize(nofAntennas);

    // Register compute kernel.
    if (outputPhasorTensor.dtype() == DataType::CF32) {
        kernel = [this]() { return kernelCF32(); };
        return Result::SUCCESS;
    }
    
    JST_ERROR("[MODULE_PHASOR_NATIVE_CPU] Unsupported output "
              "data type: {}.", outputPhasorTensor.dtype());
    return Result::ERROR;
}

Result PhasorImplNativeCpu::computeSubmit() {
    return kernel();
}

template<typename T>
static Result phasorKernel(const Tensor& antennaPositionTensor,
                           const Tensor& antennaCalibrationTensor,
                           const Tensor& boresightCoordinateTensor,
                           const Tensor& beamCoordinateTensor,
                           const Tensor& julianDateTensor,
                           const Tensor& dut1Tensor,
                                 std::vector<Triplet>& antennasXyz,
                                 std::vector<Triplet>& boresightUvw,
                                 std::vector<Triplet>& sourceUvw,
                                 std::vector<F64>& boresightDelay,
                                 F64 observationFrequencyHz,
                                 F64 channelBandwidthHz,
                                 F64 totalBandwidthHz,
                                 U64 frequencyStartIndex,
                                 U64 referenceAntennaIndex,
                                 F64 arrayReferenceLongitude,
                                 F64 arrayReferenceLatitude,
                                 F64 arrayReferenceAltitude,
                                 Tensor& outputDelayTensor,
                                 Tensor& outputPhasorTensor) {

    const Shape phasorShape = outputPhasorTensor.shape();
    const U64 nofBeams = phasorShape[0];
    const U64 nofAntennas = phasorShape[1];
    const U64 nofChannels = phasorShape[2];
    const U64 nofPolarizations = phasorShape[3];
    
    // Translate Earth Centered XYZ Antenna Coordinates (ECEF)
    // relative to Reference Position (XYZ).
    // Copy the antenna positions as they are translated in place.
    std::memcpy(
        antennasXyz.data(),
        antennaPositionTensor.data(),
        antennaPositionTensor.size() * sizeof(F64));
    calc_position_to_xyz_frame_from_ecef(
        (F64*)antennasXyz.data(),
        nofAntennas,
        arrayReferenceLongitude,
        arrayReferenceLatitude,
        arrayReferenceAltitude);
        
    F64 boresight_ha = 0.0;
    F64 boresight_dec = 0.0;
    eraASTROM astrom;

    // Convert source RA & Declination to Hour Angle.
    calc_independent_astrom(
        arrayReferenceLongitude,
        arrayReferenceLatitude,
        arrayReferenceAltitude,
        ((F64*) julianDateTensor.data())[0],
        ((F64*) dut1Tensor.data())[0],
        &astrom);

    //  Convert Boresight RA & Declination to Hour Angle & Declination.
    // const auto& boresightCoord = (F64*) boresightCoordinateTensor.data();
    calc_ha_dec_rad_with_independent_astrom(
        boresightCoordinateTensor.at<F64>(0), // RA
        boresightCoordinateTensor.at<F64>(1), // DEC
        &astrom, 
        &boresight_ha,
        &boresight_dec);

    //  Copy Reference Position (XYZ) to Boresight Position (UVW).
    for (U64 i = 0; i < nofAntennas; i++) {
        boresightUvw[i] = reinterpret_cast<const Triplet&>(antennasXyz[i]);
    }

    calc_position_delays(
        (F64*)boresightUvw.data(),
        nofAntennas,
        referenceAntennaIndex,
        boresight_ha,
        boresight_dec,
        arrayReferenceLongitude,
        boresightDelay.data()
    );


    F64* outDelayPtr = outputDelayTensor.data<F64>();
    T* outPhasorPtr = outputPhasorTensor.data<T>();
    
    for (U64 b = 0; b < nofBeams; b++) {
        //  Copy Reference Position (XYZ) to Source Position (UVW).
        for (U64 i = 0; i < nofAntennas; i++) {
            sourceUvw[i] = reinterpret_cast<const Triplet&>(antennasXyz[i]);
        }

        F64 source_ha = 0.0;
        F64 source_dec = 0.0;

        //  Convert source RA & Declination to Hour Angle
        calc_ha_dec_rad_with_independent_astrom(
            beamCoordinateTensor.at<F64>(b, (U64) 0), // RA
            beamCoordinateTensor.at<F64>(b, (U64) 1), // DEC
            &astrom, 
            &source_ha,
            &source_dec);

        calc_position_delays(
            (F64*)sourceUvw.data(),
            nofAntennas,
            referenceAntennaIndex,
            source_ha,
            source_dec,
            arrayReferenceLongitude,
            outDelayPtr + (b * nofAntennas)
        );

        //  Subtract boresight (TPi = ((WPi - WPr) / C) - Ti).
        for (U64 a = 0; a < nofAntennas; a++) {
            outDelayPtr[(b * nofAntennas) + a] -= boresightDelay[a];
        }
    }

    // TODO: Implement frequency channel expansion by some ratio.
    for (U64 b = 0; b < nofBeams; b++) {
        const U64 beamOffset = (b * 
                                nofAntennas * 
                                nofChannels * 
                                nofPolarizations); 

        for (U64 a = 0; a < nofAntennas; a++) {
            const U64 antennaOffset = (a *
                                       nofChannels *
                                       nofPolarizations);

            const F64 delay = outDelayPtr[(b * nofAntennas) + a];
            const F64 fringe = observationFrequencyHz - (totalBandwidthHz / 2.0);
            const CF64 fringeRateExp(0, -2 * M_PI * delay * fringe); 

            for (U64 f = 0; f < nofChannels; f++) {
                const U64 frequencyOffset = (f * nofPolarizations);

                const F64 freq = (f + frequencyStartIndex) * channelBandwidthHz;
                const CF64 phasorsExp(0, -2 * M_PI * delay * freq); 
                const CF64 phasor = std::exp(phasorsExp + fringeRateExp);

                for (U64 p = 0; p < nofPolarizations; p++) {
                    const U64 polarizationOffset = p;

                    const U64 calibrationIndex = antennaOffset + frequencyOffset + polarizationOffset; 
                    const U64 phasorsIndex = beamOffset + calibrationIndex;

                    outPhasorPtr[phasorsIndex] = static_cast<const T>(phasor * ((CF64*)antennaCalibrationTensor.data())[calibrationIndex]);
                }
            }
        }
    }

    return Result::SUCCESS;
}

Result PhasorImplNativeCpu::kernelCF32() {
    return phasorKernel<CF32>(
        // inputs
        antennaPositionTensor,
        antennaCalibrationTensor,
        boresightCoordinateTensor,
        beamCoordinateTensor,
        julianDateTensor,
        dut1Tensor,
        // implementation internals
        antennasXyz,
        boresightUvw,
        sourceUvw,
        boresightDelay,
        // module configuration constants
        this->observationFrequencyHz,
        this->channelBandwidthHz,
        this->totalBandwidthHz,
        this->frequencyStartIndex,
        this->referenceAntennaIndex,
        this->arrayReferenceLongitude,
        this->arrayReferenceLatitude,
        this->arrayReferenceAltitude,
        // outputs
        outputDelayTensor,
        outputPhasorTensor
    );
}

JST_REGISTER_MODULE(PhasorImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
