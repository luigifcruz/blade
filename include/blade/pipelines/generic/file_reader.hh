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

    constexpr const U64 getNumberOfBeams() const {
        return bfr5->getTotalNumberOfBeams();
    }

    constexpr const U64 getNumberOfAntennas() const {
        return guppi->getNumberOfAntennas();
    }

    constexpr const U64 getNumberOfFrequencyChannels() const {
        return guppi->getNumberOfFrequencyChannels();
    }

    constexpr const U64 getNumberOfTimeSamples() const {
        return guppi->getNumberOfTimeSamples();
    }

    constexpr const U64 getNumberOfPolarizations() const {
        return guppi->getNumberOfPolarizations();
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

    constexpr const LLA getReferencePosition() const {
        return bfr5->getReferencePosition();
    }

    constexpr const RA_DEC getBoresightCoordinate() const {
        return bfr5->getBoresightCoordinate();
    }

    constexpr const std::vector<XYZ> getAntennaPositions() const {
        return bfr5->getAntennaPositions();
    }

    constexpr const std::vector<CF64> getAntennaCalibrations(const U64& preBeamformerChannelizerRate) const {
        return bfr5->getAntennaCalibrations(guppi->getNumberOfFrequencyChannels(), preBeamformerChannelizerRate);
    }

    constexpr const std::vector<RA_DEC> getBeamCoordinates() const {
        return bfr5->getBeamCoordinates();
    }

    constexpr Modules::Guppi::Reader<OT>& getGuppi() {
        return *guppi;
    }

    Result run();

    const Vector<Device::CPU, OT>& getOutput() {
        return guppi->getOutput();
    }

    const F64 getOutputEpochSeconds() {
        return guppi->getBlockEpochSeconds(getNumberOfTimeSamples() / 2); 
    }

 private:
    const Config config;

    std::shared_ptr<Modules::Guppi::Reader<OT>> guppi;
    std::shared_ptr<Modules::Bfr5::Reader> bfr5;
};

}  // namespace Blade::Pipelines::Generic

#endif

