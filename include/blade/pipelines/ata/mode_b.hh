#ifndef BLADE_PIPELINES_ATA_MODE_B_HH
#define BLADE_PIPELINES_ATA_MODE_B_HH

#include <memory>
#include <deque>

#include "blade/pipeline.hh"

#include "blade/modules/cast.hh"
#include "blade/modules/channelizer.hh"
#include "blade/modules/beamformer/ata.hh"
#include "blade/modules/phasor/ata.hh"
#include "blade/modules/detector.hh"

namespace Blade::Pipelines::ATA {

// TODO: Add input types.

template<typename OT>
class BLADE_API ModeB : public Pipeline {
 public:
    // Configuration 

    struct Config {
        ArrayTensorDimensions inputDimensions;

        U64 preBeamformerChannelizerRate;

        F64 phasorObservationFrequencyHz;
        F64 phasorChannelBandwidthHz;
        F64 phasorTotalBandwidthHz;
        U64 phasorFrequencyStartIndex;
        U64 phasorReferenceAntennaIndex;
        LLA phasorArrayReferencePosition; 
        RA_DEC phasorBoresightCoordinate;
        std::vector<XYZ> phasorAntennaPositions;
        std::vector<CF64> phasorAntennaCalibrations; 
        std::vector<RA_DEC> phasorBeamCoordinates;

        BOOL beamformerIncoherentBeam = false;

        BOOL detectorEnable = false;
        U64 detectorIntegrationSize;
        U64 detectorNumberOfOutputPolarizations;

        U64 castBlockSize = 512;
        U64 channelizerBlockSize = 512;
        U64 phasorBlockSize = 512;
        U64 beamformerBlockSize = 512;
        U64 detectorBlockSize = 512;
    };

    // Input

    const Result transferIn(const Vector<Device::CPU, F64>& blockJulianDate,
                            const Vector<Device::CPU, F64>& blockDut1,
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
                return detector->getOutputBuffer();
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

    std::shared_ptr<Modules::Cast<CI8, CF32>> inputCast;
    std::shared_ptr<Modules::Channelizer<CF32, CF32>> channelizer;
    std::shared_ptr<Modules::Phasor::ATA<CF32>> phasor;
    std::shared_ptr<Modules::Beamformer::ATA<CF32, CF32>> beamformer;
    std::shared_ptr<Modules::Detector<CF32, F32>> detector;

    // Output Cast for path without Detector (CF32).
    std::shared_ptr<Modules::Cast<CF32, OT>> complexOutputCast;
    // Output Cast for path with Detector (F32).
    std::shared_ptr<Modules::Cast<F32, OT>> outputCast;
};

}  // namespace Blade::Pipelines::ATA

#endif
