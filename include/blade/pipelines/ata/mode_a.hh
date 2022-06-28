#ifndef BLADE_PIPELINES_ATA_MODE_A_HH
#define BLADE_PIPELINES_ATA_MODE_A_HH

#include <memory>
#include <deque>

#include "blade/pipeline.hh"

#include "blade/modules/cast.hh"
#include "blade/modules/channelizer.hh"
#include "blade/modules/beamformer/ata.hh"
#include "blade/modules/phasor/ata.hh"
#include "blade/modules/detector.hh"

namespace Blade::Pipelines::ATA {

template<typename OT = F32>
class BLADE_API ModeA : public Pipeline {
 public:
    struct Config {
        U64 numberOfAntennas;
        U64 numberOfFrequencyChannels;
        U64 numberOfTimeSamples;
        U64 numberOfPolarizations;

        U64 channelizerRate;

        U64 beamformerBeams;
        BOOL enableIncoherentBeam = false;

        F64 rfFrequencyHz;
        F64 channelBandwidthHz;
        F64 totalBandwidthHz;
        U64 frequencyStartIndex;
        U64 referenceAntennaIndex;
        LLA arrayReferencePosition; 
        RA_DEC boresightCoordinate;
        std::vector<XYZ> antennaPositions;
        std::vector<CF64> antennaCalibrations; 
        std::vector<RA_DEC> beamCoordinates;

        U64 integrationSize;
        U64 numberOfOutputPolarizations;

        U64 outputMemWidth;
        U64 outputMemPad;

        U64 castBlockSize = 512;
        U64 channelizerBlockSize = 512;
        U64 phasorsBlockSize = 512;
        U64 beamformerBlockSize = 512;
        U64 detectorBlockSize = 512;
    };

    explicit ModeA(const Config& config);

    constexpr const U64 getInputSize() const {
        return channelizer->getBufferSize();
    }

    constexpr const U64 getOutputSize() const {
        return detector->getOutputSize();
    }

    Result run(const F64& frameJulianDate,
               const F64& frameDut1,
               const Vector<Device::CPU, CI8>& input,
                     Vector<Device::CPU, OT>& output);

 private:
    const Config config;

    U64 outputMemPitch;

    Vector<Device::CUDA, CI8> input;
    Vector<Device::CPU, F64> frameJulianDate;
    Vector<Device::CPU, F64> frameDut1;

    std::shared_ptr<Modules::Cast<CI8, CF32>> inputCast;
    std::shared_ptr<Modules::Channelizer<CF32, CF32>> channelizer;
    std::shared_ptr<Modules::Phasor::ATA<CF32>> phasor;
    std::shared_ptr<Modules::Beamformer::ATA<CF32, CF32>> beamformer;
    std::shared_ptr<Modules::Detector<CF32, F32>> detector;

    constexpr const Vector<Device::CUDA, OT>& getOutput() {
        return detector->getOutput();
    }
};

}  // namespace Blade::Pipelines::ATA

#endif
