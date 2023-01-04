#ifndef BLADE_PIPELINES_ATA_MODE_B_HH
#define BLADE_PIPELINES_ATA_MODE_B_HH

#include <memory>
#include <deque>

#include "blade/pipeline.hh"

#include "blade/modules/cast.hh"
#include "blade/modules/channelizer.hh"
#include "blade/modules/transposer.hh"
#include "blade/modules/beamformer/ata.hh"
#include "blade/modules/phasor/ata.hh"
#include "blade/modules/polarizer.hh"
#include "blade/modules/detector.hh"

namespace Blade::Pipelines::ATA {

// TODO: Add input types.

template<typename OT>
class BLADE_API ModeB : public Pipeline {
 public:
    // Configuration 

    struct Config {
        ArrayDimensions inputDimensions;

        U64 preBeamformerChannelizerRate;

        BOOL preBeamformerPolarizerConvertToCircular = false;

        F64 phasorObservationFrequencyHz;
        F64 phasorChannelBandwidthHz;
        F64 phasorTotalBandwidthHz;
        U64 phasorFrequencyStartIndex;
        U64 phasorReferenceAntennaIndex;
        LLA phasorArrayReferencePosition; 
        RA_DEC phasorBoresightCoordinate;
        std::vector<XYZ> phasorAntennaPositions;
        std::vector<CF64> phasorAntennaCoefficients; 
        std::vector<RA_DEC> phasorBeamCoordinates;

        BOOL beamformerIncoherentBeam = false;

        BOOL detectorEnable = false;
        U64 detectorIntegrationSize;
        U64 detectorNumberOfOutputPolarizations;
        BOOL detectorTransposedATPFrevOutput = false;

        U64 castBlockSize = 512;
        U64 channelizerBlockSize = 512;
        U64 phasorBlockSize = 512;
        U64 beamformerBlockSize = 512;
        U64 polarizerBlockSize = 512;
        U64 detectorBlockSize = 512;
    };

    // Input

    const Result transferIn(const Vector<Device::CPU, F64>& blockJulianDate,
                            const Vector<Device::CPU, F64>& blockDut1,
                            const Vector<Device::CPU, U64>& blockFrequencyChannelOffset,
                            const ArrayTensor<Device::CPU, CI8>& input,
                            const cudaStream_t& stream);

    constexpr const ArrayTensor<Device::CUDA, CI8>& getInputBuffer() const {
        return input;
    }

    // Output 

    constexpr const ArrayTensor<Device::CUDA, OT>& getOutputBuffer() {
        if (config.detectorEnable) {
            if constexpr (!std::is_same<OT, F32>::value) {
                return outputCast->getOutputBuffer();
            } else {
                if (config.detectorTransposedATPFrevOutput) {
                    return transposer->getOutputBuffer();
                } else {
                    return detector->getOutputBuffer();
                }
            }
        } else {
            if constexpr (!std::is_same<OT, CF32>::value) {
                return complexOutputCast->getOutputBuffer();
            } else {
                return beamformer->getOutputBuffer();
            }
        }
    }

    // Constructor

    explicit ModeB(const Config& config);

 private:
    const Config config;

    ArrayTensor<Device::CUDA, CI8> input;
    Vector<Device::CPU, F64> blockJulianDate;
    Vector<Device::CPU, F64> blockDut1;
    Vector<Device::CPU, U64> blockFrequencyChannelOffset;

    using InputCast = typename Modules::Cast<CI8, CF32>;
    std::shared_ptr<InputCast> inputCast;

    using PreChannelizer = typename Modules::Channelizer<CF32, CF32>;
    std::shared_ptr<PreChannelizer> channelizer;

    using Phasor = typename Modules::Phasor::ATA<CF32>;
    std::shared_ptr<Phasor> phasor;

    using Beamformer = typename Modules::Beamformer::ATA<CF32, CF32>;
    std::shared_ptr<Beamformer> beamformer;

    using Polarizer = typename Modules::Polarizer<CF32, CF32>;
    std::shared_ptr<Polarizer> polarizer;

    using Detector = typename Modules::Detector<CF32, F32>;
    std::shared_ptr<Detector> detector;

    // Transposition for path with Detector enabling output to match Filterbank dimensionality (ATPF)
    std::shared_ptr<Modules::Transposer<Device::CUDA, F32, ArrayDimensionOrder::AFTP, ArrayDimensionOrder::ATPFrev>> transposer;

    // Output Cast for path without Detector (CF32).
    std::shared_ptr<Modules::Cast<CF32, OT>> complexOutputCast;
    // Output Cast for path with Detector (F32).
    std::shared_ptr<Modules::Cast<F32, OT>> outputCast;
};

}  // namespace Blade::Pipelines::ATA

#endif
