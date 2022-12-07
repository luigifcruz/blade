#ifndef BLADE_PIPELINES_GENERIC_MODE_H_HH
#define BLADE_PIPELINES_GENERIC_MODE_H_HH

#include <memory>
#include <deque>

#include "blade/pipeline.hh"

#include "blade/modules/channelizer.hh"
#include "blade/modules/detector.hh"
#include "blade/modules/cast.hh"
#include "blade/modules/polarizer.hh"

namespace Blade::Pipelines::Generic {

template<typename IT, typename OT>
class BLADE_API ModeH : public Pipeline {
 public:
    // Configuration 

    struct Config {
        ArrayDimensions inputDimensions;

        U64 accumulateRate;

        BOOL polarizerConvertToCircular = false;

        U64 detectorIntegrationSize;
        U64 detectorNumberOfOutputPolarizations;

        U64 castBlockSize = 512;
        U64 polarizerBlockSize = 512;
        U64 channelizerBlockSize = 512;
        U64 detectorBlockSize = 512;
    };

    // Input

    const Result accumulate(const ArrayTensor<Device::CUDA, IT>& data,
                            const cudaStream_t& stream);

    constexpr const ArrayTensor<Device::CUDA, IT>& getInputBuffer() const {
        return input;
    }

    // Output 

    constexpr const ArrayTensor<Device::CUDA, OT>& getOutputBuffer() {
        return detector->getOutputBuffer();
    }

    // Constructor

    explicit ModeH(const Config& config);

 private:
    const Config config;

    ArrayTensor<Device::CUDA, IT> input;

    using InputCast = typename Modules::Cast<CF16, CF32>;
    std::shared_ptr<InputCast> cast;

    using PreChannelizer = typename Modules::Channelizer<CF32, CF32>;
    std::shared_ptr<PreChannelizer> channelizer;

    using Polarizer = typename Modules::Polarizer<CF32, CF32>;
    std::shared_ptr<Polarizer> polarizer;

    using Detector = typename Modules::Detector<CF32, F32>;
    std::shared_ptr<Detector> detector;

    constexpr const ArrayTensor<Device::CUDA, CF32>& getChannelizerInput() {
        if constexpr (!std::is_same<IT, CF32>::value) {
            return cast->getOutputBuffer();
        } else {
            return input;
        }
    }
};

}  // namespace Blade::Pipelines::Generic

#endif
