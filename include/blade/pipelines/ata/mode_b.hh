#ifndef BLADE_PIPELINES_ATA_MODE_B_HH
#define BLADE_PIPELINES_ATA_MODE_B_HH

#include <memory>
#include <deque>

#include "blade/pipeline.hh"

#include "blade/modules/cast.hh"
#include "blade/modules/channelizer.hh"
#include "blade/modules/beamformer/ata.hh"
extern "C" {
#include "mode_b_config.h"
}

namespace Blade::Pipelines::ATA {

class ModeB : public Pipeline {
 public:
    struct Config {
        ArrayDims inputDims;
        std::size_t channelizerRate = BLADE_ATA_MODE_B_CHANNELISER_RATE;
        std::size_t beamformerBeams = BLADE_ATA_MODE_B_OUTPUT_NBEAM;

        std::size_t castBlockSize = 512;
        std::size_t channelizerBlockSize = 512;
        std::size_t beamformerBlockSize = 512;
    };

    explicit ModeB(const Config& config);

    const std::size_t getInputSize() const {
        #if BLADE_ATA_MODE_B_CHANNELISER_RATE > 1
        return channelizer->getBufferSize();
        #else
        return beamformer->getInputSize();
        #endif
    }

    const std::size_t getPhasorsSize() const {
        return beamformer->getPhasorsSize();
    }

    const std::size_t getOutputSize() const {
        // include padding
        return (
            (
                (beamformer->getOutputSize()*sizeof(BLADE_ATA_MODE_B_OUTPUT_ELEMENT_T))
                /BLADE_ATA_MODE_B_OUTPUT_MEMCPY2D_WIDTH
            )*BLADE_ATA_MODE_B_OUTPUT_MEMCPY2D_DPITCH
        )/sizeof(BLADE_ATA_MODE_B_OUTPUT_ELEMENT_T);
    }

    Result run(const Vector<Device::CPU, CI8>& input,
                     Vector<Device::CPU, BLADE_ATA_MODE_B_OUTPUT_ELEMENT_T>& output);

 private:
    const Config config;

    Vector<Device::CUDA, CI8> input;
    Vector<Device::CUDA, CF32> phasors;
    Vector<Device::CUDA, BLADE_ATA_MODE_B_OUTPUT_ELEMENT_T> output;

    std::shared_ptr<Modules::Cast<CI8, CF32>> inputCast;
    #if BLADE_ATA_MODE_B_CHANNELISER_RATE > 1
    std::shared_ptr<Modules::Channelizer<CF32, CF32>> channelizer;
    #endif
    std::shared_ptr<Modules::Beamformer::ATA<CF32, CF32>> beamformer;
    #if BLADE_ATA_MODE_B_OUTPUT_NCOMPLEX_BYTES != 8
    std::shared_ptr<Modules::Cast<CF32, BLADE_ATA_MODE_B_OUTPUT_ELEMENT_T>> outputCast;
    #endif
};

}  // namespace Blade::Pipelines::ATA

#endif
