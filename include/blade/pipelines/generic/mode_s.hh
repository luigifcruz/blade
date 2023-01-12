#ifndef BLADE_PIPELINES_GENERIC_MODE_S_HH
#define BLADE_PIPELINES_GENERIC_MODE_S_HH

#include <memory>
#include <deque>

#include "blade/pipeline.hh"

#include "blade/modules/seticore/dedoppler.hh"
#include "blade/modules/seticore/hits_writer.hh"

namespace Blade::Pipelines::Generic {

class BLADE_API ModeS : public Pipeline {
 public:
    // Configuration 

    struct Config {
        ArrayDimensions prebeamformerInputDimensions;
        ArrayDimensions inputDimensions;

        U64 inputCoarseChannelRate = 1;
        BOOL inputLastBeamIsIncoherent = false;

        BOOL searchMitigateDcSpike;
        F64 searchMinimumDriftRate = 0.0;
        F64 searchMaximumDriftRate;
        F64 searchSnrThreshold;

        F64 searchChannelBandwidthHz;
        F64 searchChannelTimespanS;
        std::string searchOutputFilepathStem;

        U64 dedopplerBlockSize = 512;
    };

    // Input

    void setFrequencyOfFirstInputChannel(F64 hz);

    const Result accumulate(const ArrayTensor<Device::CUDA, F32>& data,
                            const ArrayTensor<Device::CPU, CF32>& prebeamformerData,
                            const Vector<Device::CPU, U64>& coarseFrequencyChannelOffset,
                            const cudaStream_t& stream);

    const Result accumulate(const ArrayTensor<Device::CUDA, F32>& data,
                            const ArrayTensor<Device::CUDA, CF32>& prebeamformerData,
                            const Vector<Device::CPU, U64>& coarseFrequencyChannelOffset,
                            const cudaStream_t& stream);

    // Output

    const std::vector<DedopplerHit>& getOutputHits() {
        return dedoppler->getOutputHits();
    }

    // Constructor

    explicit ModeS(const Config& config);

 private:
    const Config config;

    ArrayTensor<Device::CUDA, F32> input;
    ArrayTensor<Device::CPU, CF32> prebeamformerData;
    Vector<Device::CPU, U64> coarseFrequencyChannelOffset;

    std::shared_ptr<Modules::Seticore::Dedoppler> dedoppler;
    std::shared_ptr<Modules::Seticore::HitsWriter<CF32>> hitsWriter;

};

}  // namespace Blade::Pipelines::Generic

#endif
