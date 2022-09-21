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
    constexpr const U64 getStepNumberOfAntennas() const {
        return guppi->getStepNumberOfAntennas();
    }

    constexpr const U64 getStepNumberOfFrequencyChannels() const {
        return guppi->getStepNumberOfFrequencyChannels();
    }

    constexpr const U64 getStepNumberOfTimeSamples() const {
        return guppi->getStepNumberOfTimeSamples();
    }

    constexpr const U64 getStepNumberOfPolarizations() const {
        return guppi->getStepNumberOfPolarizations();
    }

    constexpr const U64 getStepOutputBufferSize() const {
        return guppi->getStepOutputBufferSize();
    }

    constexpr const U64 getTotalNumberOfAntennas() const {
        return guppi->getTotalNumberOfAntennas();
    }

    constexpr const U64 getTotalNumberOfFrequencyChannels() const {
        return guppi->getTotalNumberOfFrequencyChannels();
    }

    constexpr const U64 getTotalNumberOfTimeSamples() const {
        return guppi->getTotalNumberOfTimeSamples();
    }

    constexpr const U64 getTotalNumberOfPolarizations() const {
        return guppi->getTotalNumberOfPolarizations();
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

    constexpr const U64 getTotalOutputBufferSize() const {
        return guppi->getTotalOutputBufferSize();
    }

    const ArrayTensor<Device::CPU, OT>& getStepOutputBuffer() {
        return guppi->getStepOutputBuffer();
    }

    const Vector<Device::CPU, F64>& getStepOutputJulianDate() {
        return guppi->getStepOutputJulianDate();
    }

    const Vector<Device::CPU, F64>& getStepOutputDut1() {
        return guppi->getStepOutputDut1();
    }

    // BFR5 determined values
    constexpr const U64 getStepNumberOfBeams() const {
        return bfr5->getTotalNumberOfBeams();
    }

    constexpr const U64 getTotalNumberOfBeams() const {
        return bfr5->getTotalNumberOfBeams();
    }

    constexpr const LLA getReferencePosition() const {
        return bfr5->getReferencePosition();
    }

    constexpr const RA_DEC getBoresightCoordinate() const {
        return bfr5->getBoresightCoordinate();
    }

    constexpr const std::vector<XYZ> getAntennaPositions() const {
        return bfr5->getAntennaPositions();
    }

    constexpr void fillAntennaCalibrations(const U64& preBeamformerChannelizerRate, ArrayCoefficientTensor<Device::CPU, CF64>& antennaCalibrations) const {
        return bfr5->fillAntennaCalibrations(guppi->getStepNumberOfFrequencyChannels(), preBeamformerChannelizerRate, antennaCalibrations);
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

