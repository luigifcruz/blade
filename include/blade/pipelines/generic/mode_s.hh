#ifndef BLADE_PIPELINES_GENERIC_MODE_S_HH
#define BLADE_PIPELINES_GENERIC_MODE_S_HH

#include <memory>
#include <deque>

#include "blade/pipeline.hh"

#include "blade/modules/seticore/dedoppler.hh"

namespace Blade::Pipelines::Generic {

class BLADE_API ModeS : public Pipeline {
 public:
    // Configuration 

    struct Config {
        ArrayDimensions inputDimensions;
        U64 accumulateRate;

        BOOL searchMitigateDcSpike;
        F64 searchMinimumDriftRate = 0.0;
        F64 searchMaximumDriftRate;
        F64 searchSnrThreshold;

        F64 searchChannelBandwidthHz;
        F64 searchChannelTimespanS;

        U64 dedopplerBlockSize = 512;
    };

    // Input

    const Result accumulate(const ArrayTensor<Device::CUDA, F32>& data,
                            const cudaStream_t& stream);

    constexpr const ArrayTensor<Device::CUDA, F32>& getInputBuffer() const {
        return input;
    }

    // Output

    const std::vector<DedopplerHit>& getOutputHits() {
        return dedoppler->getOutputHits();
    }

    // Constructor

    explicit ModeS(const Config& config);

 private:
    const Config config;

    ArrayTensor<Device::CUDA, F32> input;

    std::shared_ptr<Modules::Seticore::Dedoppler> dedoppler;
};

}  // namespace Blade::Pipelines::Generic

#endif
