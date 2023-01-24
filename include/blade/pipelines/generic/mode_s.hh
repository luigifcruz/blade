#ifndef BLADE_PIPELINES_GENERIC_MODE_S_HH
#define BLADE_PIPELINES_GENERIC_MODE_S_HH

#include <memory>
#include <deque>

#include "blade/pipeline.hh"

#include "blade/modules/seticore/dedoppler.hh"
#include "blade/modules/seticore/hits_raw_writer.hh"
#include "blade/modules/seticore/hits_stamp_writer.hh"

namespace Blade::Pipelines::Generic {

enum class BLADE_API HitsFormat : uint8_t {
    NONE            = 0,
    GUPPI_RAW       = 1,
    SETICORE_STAMP  = 2
};

constexpr const char* HitsFormatName(const HitsFormat order) {
    switch (order) {
        case HitsFormat::GUPPI_RAW:
            return "GUPPI RAW";
        case HitsFormat::SETICORE_STAMP:
            return "SETICORE STAMP";
        default:
            return "None";
    }
}

template<HitsFormat HT>
class BLADE_API ModeS : public Pipeline {
 public:
    // Configuration 

    struct Config {
        ArrayDimensions prebeamformerInputDimensions;
        ArrayDimensions inputDimensions;

        U64 inputTelescopeId;
        std::string inputSourceName;
        std::string inputObservationIdentifier;
        RA_DEC inputPhaseCenter;
        U64 inputTotalNumberOfTimeSamples;
        U64 inputTotalNumberOfFrequencyChannels;
        U64 inputCoarseStartChannelIndex;
        F64 inputJulianDateStart;
        U64 inputCoarseChannelRatio = 1;
        BOOL inputLastBeamIsIncoherent = false;

        std::vector<std::string> beamNames;
        std::vector<RA_DEC> beamCoordinates;

        BOOL searchMitigateDcSpike;
        F64 searchMinimumDriftRate = 0.0;
        F64 searchMaximumDriftRate;
        F64 searchSnrThreshold;

        F64 searchChannelBandwidthHz;
        F64 searchChannelTimespanS;
        std::string searchOutputFilepathStem;

        U64 dedopplerBlockSize = 512;
    };

    // Input

    void setFrequencyOfFirstInputChannel(F64 hz);

    const Result accumulate(const ArrayTensor<Device::CUDA, F32>& data,
                            const ArrayTensor<Device::CPU, CF32>& prebeamformerData,
                            const Vector<Device::CPU, U64>& coarseFrequencyChannelOffset,
                            const cudaStream_t& stream);

    const Result accumulate(const ArrayTensor<Device::CUDA, F32>& data,
                            const ArrayTensor<Device::CUDA, CF32>& prebeamformerData,
                            const Vector<Device::CPU, U64>& coarseFrequencyChannelOffset,
                            const cudaStream_t& stream);

    // Output

    const std::vector<DedopplerHit>& getOutputHits() {
        return dedoppler->getOutputHits();
    }

    // Constructor

    explicit ModeS(const Config& config);

 private:
    const Config config;

    ArrayTensor<Device::CUDA, F32> input;
    ArrayTensor<Device::CPU, CF32> prebeamformerData;
    Vector<Device::CPU, U64> coarseFrequencyChannelOffset;
    Vector<Device::CPU, F64> frequencyOfFirstInputChannelHz;

    std::shared_ptr<Modules::Seticore::Dedoppler> dedoppler;
    std::shared_ptr<Modules::Seticore::HitsRawWriter<CF32>> hitsRawWriter;
    std::shared_ptr<Modules::Seticore::HitsStampWriter<CF32>> hitsStampWriter;

};

}  // namespace Blade::Pipelines::Generic

#endif
