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

    constexpr const ArrayTensorDimensions getAntennaCalibrationsDims(const U64& channelizerRate) const {
        return bfr5->getAntennaCalibrationsDims(channelizerRate);
    }

    constexpr void fillAntennaCalibrations(const U64& preBeamformerChannelizerRate, ArrayTensor<Device::CPU, CF64>& antennaCalibrations) const {
        return bfr5->fillAntennaCalibrations(preBeamformerChannelizerRate, antennaCalibrations);
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

