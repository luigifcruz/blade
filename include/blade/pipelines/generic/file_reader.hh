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

        U64 channelizerRate;
        U64 stepNumberOfTimeSamples;
        U64 stepNumberOfFrequencyChannels;
    };

    explicit FileReader(const Config& config);

    // GUPPI RAW determined values
    constexpr const ArrayShape getStepOutputShape() const {
        return guppi->getStepOutputBufferShape();
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

    constexpr const ArrayShape getTotalOutputShape() const {
        return guppi->getTotalOutputBufferShape();
    }

    constexpr const U64 getTotalOutputBufferSize() const {
        return guppi->getTotalOutputBufferShape().size();
    }

    const ArrayTensor<Device::CPU, OT>& getStepOutputBuffer() {
        return guppi->getStepOutputBuffer();
    }

    const Tensor<Device::CPU, F64>& getStepOutputJulianDate() {
        return guppi->getStepOutputJulianDate();
    }

    const Tensor<Device::CPU, F64>& getStepOutputDut1() {
        return guppi->getStepOutputDut1();
    }

    // BFR5 determined values

    constexpr const LLA getReferencePosition() const {
        return bfr5->getReferencePosition();
    }

    constexpr const RA_DEC getBoresightCoordinates() const {
        return bfr5->getBoresightCoordinates();
    }

    constexpr const std::vector<XYZ>& getAntennaPositions() const {
        return bfr5->getAntennaPositions();
    }

    constexpr const std::vector<RA_DEC>& getBeamCoordinates() const {
        return bfr5->getBeamCoordinates();
    }

    constexpr const ArrayTensor<Device::CPU, CF64>& getAntennaCalibrations() const {
        return bfr5->getAntennaCalibrations();
    }

 private:
    const Config config;

    using GuppiReader = typename Modules::Guppi::Reader<OT>;
    std::shared_ptr<Modules::Guppi::Reader<OT>> guppi;

    using Bfr5Reader = typename Modules::Bfr5::Reader; 
    std::shared_ptr<Bfr5Reader> bfr5;
};

}  // namespace Blade::Pipelines::Generic

#endif

