#ifndef BLADE_PIPELINES_VLA_MODE_B_HH
#define BLADE_PIPELINES_VLA_MODE_B_HH

#include <memory>
#include <deque>

#include "blade/pipeline.hh"

#include "blade/modules/cast.hh"
#include "blade/modules/channelizer.hh"
#include "blade/modules/beamformer/meerkat.hh"
#include "blade/modules/detector.hh"

namespace Blade::Pipelines::VLA {

template<typename IT, typename OT>
class BLADE_API ModeB : public Pipeline {
 public:
    // Configuration 

    struct Config {
        ArrayTensorDimensions inputDimensions;

        U64 preBeamformerChannelizerRate;

        PhasorTensor<Device::CPU, CF32> beamformerPhasors;
        U64 beamformerNumberOfBeams;
        BOOL beamformerIncoherentBeam = false;

        BOOL detectorEnable = false;
        U64 detectorIntegrationSize;
        U64 detectorNumberOfOutputPolarizations;

        U64 castBlockSize = 512;
        U64 channelizerBlockSize = 512;
        U64 beamformerBlockSize = 512;
        U64 detectorBlockSize = 512;
    };

    // Input

    const Result transferIn(const ArrayTensor<Device::CPU, IT>& input,
                            const cudaStream_t& stream);

    constexpr const ArrayTensor<Device::CUDA, IT>& getInputBuffer() const {
        return input;
    }

    // Output 

    constexpr const ArrayTensor<Device::CUDA, OT>& getOutputBuffer() {
        if (config.detectorEnable) {
            if constexpr (!std::is_same<OT, F32>::value) {
                return outputCast->getOutputBuffer();
            } else {
                return detector->getOutputBuffer();
            }
        } else {
            if constexpr (!std::is_same<OT, CF32>::value) {
                return complexOutputCast->getOutputBuffer();
            } else {
                return beamformer->getOutputBuffer();
            }
        }
    }

    // Constructor

    explicit ModeB(const Config& config);

 private:
    const Config config;

    ArrayTensor<Device::CUDA, IT> input;
    PhasorTensor<Device::CUDA, CF32> phasors;

    std::shared_ptr<Modules::Cast<IT, CF32>> inputCast;
    std::shared_ptr<Modules::Channelizer<CF32, CF32>> channelizer;
    std::shared_ptr<Modules::Beamformer::MeerKAT<CF32, CF32>> beamformer;
    std::shared_ptr<Modules::Detector<CF32, F32>> detector;

    // Output Cast for path without Detector (CF32).
    std::shared_ptr<Modules::Cast<CF32, OT>> complexOutputCast;
    // Output Cast for path with Detector (F32).
    std::shared_ptr<Modules::Cast<F32, OT>> outputCast;
};

}  // namespace Blade::Pipelines::VLA

#endif
