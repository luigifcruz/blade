#ifndef BLADE_PIPELINES_ATA_MODE_B_HH
#define BLADE_PIPELINES_ATA_MODE_B_HH

#include <memory>
#include <deque>

#include "blade/pipeline.hh"

#include "blade/modules/cast.hh"
#include "blade/modules/channelizer.hh"
#include "blade/modules/beamformer/ata.hh"

namespace Blade::Pipelines::ATA {

template<typename OT = CF16>
class ModeB : public Pipeline {
 public:
    struct Config {
        ArrayDims inputDims;
        std::size_t channelizerRate;  // 1 mitigates the channelization
        std::size_t beamformerBeams;

        std::size_t outputMemWidth;
        std::size_t outputMemPad;

        std::size_t castBlockSize = 512;
        std::size_t channelizerBlockSize = 512;
        std::size_t beamformerBlockSize = 512;
    };

    explicit ModeB(const Config& config);

    constexpr const std::size_t getInputSize() const {
        if (config.channelizerRate > 1) {
            return channelizer->getBufferSize();
        } else {
            return beamformer->getInputSize();
        }
    }

    constexpr const std::size_t getPhasorsSize() const {
        return beamformer->getPhasorsSize();
    }

    constexpr const std::size_t getOutputSize() const {
        return
            (((beamformer->getOutputSize() * sizeof(OT)) / config.outputMemWidth) *
                outputMemPitch) / sizeof(OT);
    }

    Result run(const Vector<Device::CPU, CI8>& input,
                     Vector<Device::CPU, OT>& output);

 private:
    const Config config;

    std::size_t outputMemPitch;

    Vector<Device::CUDA, CI8> input;
    Vector<Device::CUDA, CF32> phasors;

    std::shared_ptr<Modules::Cast<CI8, CF32>> inputCast;
    std::shared_ptr<Modules::Channelizer<CF32, CF32>> channelizer;
    std::shared_ptr<Modules::Beamformer::ATA<CF32, CF32>> beamformer;
    std::shared_ptr<Modules::Cast<CF32, OT>> outputCast;

    constexpr const Vector<Device::CUDA, OT>& getOutput() {
        if constexpr (!std::is_same<OT, CF32>::value) {
            // output is casted output
            return outputCast->getOutput();
        } else {
            // output is un-casted beamformer output (CF32)
            return beamformer->getOutput();
        }
    }
};

}  // namespace Blade::Pipelines::ATA

#endif
