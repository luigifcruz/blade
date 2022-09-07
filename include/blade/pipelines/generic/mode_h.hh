#ifndef BLADE_PIPELINES_GENERIC_MODE_H_HH
#define BLADE_PIPELINES_GENERIC_MODE_H_HH

#include <memory>
#include <deque>

#include "blade/pipeline.hh"
#include "blade/accumulator.hh"

#include "blade/modules/channelizer.hh"
#include "blade/modules/detector.hh"
#include "blade/modules/cast.hh"

namespace Blade::Pipelines::Generic {

template<typename IT, typename OT>
class BLADE_API ModeH : public Pipeline, public Accumulator {
 public:
    struct Config {
        U64 accumulateRate;

        U64 channelizerNumberOfBeams;
        U64 channelizerNumberOfFrequencyChannels;
        U64 channelizerNumberOfTimeSamples;
        U64 channelizerNumberOfPolarizations;

        U64 detectorNumberOfOutputPolarizations;

        U64 castBlockSize = 512;
        U64 channelizerBlockSize = 512;
        U64 detectorBlockSize = 512;
    };

    explicit ModeH(const Config& config);

    constexpr const U64 getInputSize() const {
        return channelizer->getBufferSize();
    }

    const Result accumulate(const ArrayTensor<Device::CUDA, IT>& data,
                            const cudaStream_t& stream);

    constexpr const U64 getOutputSize() const {
        return detector->getOutputSize();
    }

    constexpr const ArrayTensor<Device::CUDA, OT>& getOutput() {
        return detector->getOutput();
    }

 private:
    const Config config;

    ArrayTensor<Device::CUDA, IT> input;

    std::shared_ptr<Modules::Cast<CF16, CF32>> cast;
    std::shared_ptr<Modules::Channelizer<CF32, CF32>> channelizer;
    std::shared_ptr<Modules::Detector<CF32, F32>> detector;

    constexpr const ArrayTensor<Device::CUDA, CF32>& getChannelizerInput() {
        if constexpr (!std::is_same<IT, CF32>::value) {
            return cast->getOutput();
        } else {
            return input;
        }
    }
};

}  // namespace Blade::Pipelines::Generic

#endif
