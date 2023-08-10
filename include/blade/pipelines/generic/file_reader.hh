#ifndef BLADE_PIPELINES_GENERIC_FILE_READER_HH
#define BLADE_PIPELINES_GENERIC_FILE_READER_HH

#include <memory>
#include <deque>

#include "blade/pipeline.hh"

#include "blade/modules/guppi/reader.hh"
#include "blade/modules/bfr5/reader.hh"

namespace Blade::Pipelines::Generic {

template<typename OT = CI8>
class BLADE_API FileReader : public Pipeline {
 public:
    struct Config {
        std::string inputGuppiFile;
        std::string inputBfr5File;

        U64 stepNumberOfTimeSamples;
        U64 requiredMultipleOfTimeSamplesSteps = 1;
        U64 stepNumberOfFrequencyChannels;
        U64 numberOfTimeSampleStepsBeforeFrequencyChannelStep = 1;
        U64 numberOfGuppiFilesLimit = 0; // zero for no limit
    };

    explicit FileReader(const Config& config);

    // GUPPI RAW determined values
    constexpr const ArrayDimensions getStepOutputDims() const {
        return guppi->getStepOutputBufferDims();
    }

    constexpr const U64 getStepOutputBufferSize() const {
        return guppi->getStepOutputBuffer().size();
    }

    constexpr const F64 getObservationCenterFrequency() const {
        return guppi->getObservationCenterFrequency();
    }

    constexpr const F64 getCenterFrequency() const {
        return guppi->getCenterFrequency();
    }

    constexpr const F64 getObservationBottomFrequency() const {
        return guppi->getObservationBottomFrequency();
    }

    constexpr const F64 getBottomFrequency() const {
        return guppi->getBottomFrequency();
    }

    constexpr const F64 getObservationTopFrequency() const {
        return guppi->getObservationTopFrequency();
    }

    constexpr const F64 getTopFrequency() const {
        return guppi->getTopFrequency();
    }

    constexpr const F64 getChannelBandwidth() const {
        return guppi->getChannelBandwidth();
    }

    constexpr const F64 getChannelTimespan() const {
        return guppi->getChannelTimespan();
    }

    constexpr const F64 getObservationBandwidth() const {
        return guppi->getObservationBandwidth();
    }

    constexpr const F64 getBandwidth() const {
        return guppi->getBandwidth();
    }

    constexpr const U64 getChannelStartIndex() const {
        return guppi->getChannelStartIndex();
    }
    
    constexpr const F64 getAzimuthAngle() const {
        return guppi->getAzimuthAngle();
    }

    constexpr const F64 getZenithAngle() const {
        return guppi->getZenithAngle();
    }

    constexpr const std::string getSourceName() const {
        return guppi->getSourceName();
    }

    constexpr const std::string getTelescopeName() const {
        return guppi->getTelescopeName();
    }

    constexpr const U64 getNumberOfSteps() const {
        return guppi->getNumberOfSteps();
    }

    constexpr const ArrayDimensions getNumberOfStepsInDimensions() const {
        return guppi->getNumberOfStepsInDimensions();
    }

    constexpr const ArrayDimensions getTotalOutputDims() const {
        return guppi->getTotalOutputBufferDims();
    }

    constexpr const U64 getTotalOutputBufferSize() const {
        return guppi->getTotalOutputBufferDims().size();
    }

    const ArrayTensor<Device::CPU, OT>& getStepOutputBuffer() {
        return guppi->getStepOutputBuffer();
    }

    F64 getJulianDateOfLastReadBlock() const {
        return guppi->getStepOutputJulianDate()[0];
    }

    const Vector<Device::CPU, F64>& getStepOutputJulianDate() {
        return guppi->getStepOutputJulianDate();
    }

    const Vector<Device::CPU, F64>& getStepOutputDut1() {
        return guppi->getStepOutputDut1();
    }

    const Vector<Device::CPU, U64>& getStepOutputFrequencyChannelOffset() {
        return guppi->getStepOutputFrequencyChannelOffset();
    }

    // BFR5 determined values

    constexpr const LLA getReferencePosition() const {
        return bfr5->getReferencePosition();
    }

    constexpr const RA_DEC getPhaseCenterCoordinates() const {
        return bfr5->getPhaseCenterCoordinates();
    }

    constexpr const std::vector<XYZ>& getAntennaPositions() const {
        return bfr5->getAntennaPositions();
    }

    constexpr std::vector<F64> getBeamAntennaDelays() const {
        return bfr5->getBeamAntennaDelays();
    }

    constexpr std::vector<F64> getDelayTimes() const {
        return bfr5->getDelayTimes();
    }

    constexpr std::vector<CF64> getAntennaCoefficients(const U64& numberOfFrequencyChannels = 0, const U64& frequencyChannelStartIndex = 0) const {
        return bfr5->getAntennaCoefficients(numberOfFrequencyChannels, frequencyChannelStartIndex);
    }

    constexpr const std::vector<RA_DEC>& getBeamCoordinates() const {
        return bfr5->getBeamCoordinates();
    }

    constexpr const std::vector<std::string>& getBeamSourceNames() const {
        return bfr5->getBeamSourceNames();
    }

    constexpr const PhasorDimensions getRecipeDims() const {
        return bfr5->getDims();
    }

 private:
    const Config config;

    using GuppiReader = typename Modules::Guppi::Reader<OT>;
    std::shared_ptr<Modules::Guppi::Reader<OT>> guppi;

    std::shared_ptr<Modules::Bfr5::Reader> bfr5;
};

}  // namespace Blade::Pipelines::Generic

#endif

