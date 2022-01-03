#ifndef BLADE_PIPELINES_ATA_MODE_B_HH
#define BLADE_PIPELINES_ATA_MODE_B_HH

#include <memory>
#include <deque>

#include "blade/pipeline.hh"

#include "blade/modules/cast.hh"
#include "blade/modules/channelizer.hh"
#include "blade/modules/beamformer/ata.hh"

namespace Blade::Pipelines::ATA {

class ModeB : public Pipeline {
 public:
    struct Config {
        ArrayDims inputDims;
        std::size_t channelizerRate = 4;
        std::size_t beamformerBeams = 16;

        std::size_t castBlockSize = 512;
        std::size_t channelizerBlockSize = 512;
        std::size_t beamformerBlockSize = 512;
    };

    explicit ModeB(const Config& config);

    const std::size_t getInputSize() const {
        return channelizer->getBufferSize();
    }

    const std::size_t getPhasorsSize() const {
        return beamformer->getPhasorsSize();
    }

    const std::size_t getOutputSize() const {
        return beamformer->getOutputSize();
    }

    Result run(const Vector<Device::CPU, CI8>& input,
                     Vector<Device::CPU, CF16>& output);

 private:
    const Config config;

    Vector<Device::CUDA, CI8> input;
    Vector<Device::CUDA, CF32> phasors;
    Vector<Device::CUDA, CF16> output;

    std::shared_ptr<Modules::Cast<CI8, CF32>> inputCast;
    std::shared_ptr<Modules::Channelizer<CF32, CF32>> channelizer;
    std::shared_ptr<Modules::Beamformer::ATA<CF32, CF32>> beamformer;
    std::shared_ptr<Modules::Cast<CF32, CF16>> outputCast;
};

}  // namespace Blade::Pipelines::ATA

#endif
