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
        U64 stepNumberOfFrequencyChannels;
    };

    explicit FileReader(const Config& config);

    // GUPPI RAW determined values
    constexpr const ArrayTensorDimensions getStepOutputDims() const {
        return guppi->getStepOutputBufferDims();
    }

    constexpr const U64 getStepOutputBufferSize() const {
        return guppi->getStepOutputBuffer().size();
    }

    constexpr const F64 getObservationFrequency() const {
        return guppi->getObservationFrequency();
    }

    constexpr const F64 getChannelBandwidth() const {
        return guppi->getChannelBandwidth();
    }

    constexpr const F64 getTotalBandwidth() const {
        return guppi->getTotalBandwidth();
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

    constexpr const ArrayTensorDimensions getTotalOutputDims() const {
        return guppi->getTotalOutputBufferDims();
    }

    constexpr const U64 getTotalOutputBufferSize() const {
        return guppi->getTotalOutputBufferDims().size();
    }

    const ArrayTensor<Device::CPU, OT>& getStepOutputBuffer() {
        return guppi->getStepOutputBuffer();
    }

    F64 getJulianDateOfLastReadBlock() const {
        return guppi->getJulianDateOfLastReadBlock();
    }

    const Vector<Device::CPU, F64>& getStepOutputJulianDate() {
        return guppi->getStepOutputJulianDate();
    }

    const Vector<Device::CPU, F64>& getStepOutputDut1() {
        return guppi->getStepOutputDut1();
    }

    // BFR5 determined values

    constexpr const LLA getReferencePosition() const {
        return bfr5->getReferencePosition();
    }

    constexpr const RA_DEC getBoresightCoordinate() const {
        return bfr5->getBoresightCoordinate();
    }

    constexpr const std::vector<XYZ> getAntennaPositions() const {
        return bfr5->getAntennaPositions();
    }

    constexpr const PhasorTensorDimensions getBeamAntennaDelayDims() const { 
        return bfr5->getBeamAntennaDelayDims();
    }

    constexpr F64* getBeamAntennaDelays() const {
        return bfr5->getBeamAntennaDelays();
    }

    constexpr const U64 getNumberOfDelayTimes() const { 
        return bfr5->getNumberOfDelayTimes();
    }

    constexpr F64* getDelayTimes() const {
        return bfr5->getDelayTimes();
    }

    constexpr const ArrayTensorDimensions getAntennaCoefficientsDims(const U64& channelizerRate) const {
        return bfr5->getAntennaCoefficientsDims(channelizerRate);
    }

    constexpr void fillAntennaCoefficients(const U64& preBeamformerChannelizerRate, ArrayTensor<Device::CPU, CF64>& antennaCoefficients) const {
        return bfr5->fillAntennaCoefficients(preBeamformerChannelizerRate, antennaCoefficients);
    }

    constexpr const std::vector<RA_DEC> getBeamCoordinates() const {
        return bfr5->getBeamCoordinates();
    }

 private:
    const Config config;

    std::shared_ptr<Modules::Guppi::Reader<OT>> guppi;
    std::shared_ptr<Modules::Bfr5::Reader> bfr5;
};

}  // namespace Blade::Pipelines::Generic

#endif

