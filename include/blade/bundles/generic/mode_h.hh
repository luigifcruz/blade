#ifndef BLADE_BUNDLES_GENERIC_MODE_H_HH
#define BLADE_BUNDLES_GENERIC_MODE_H_HH

#include "blade/bundle.hh"

#include "blade/modules/channelizer/base.hh"
#include "blade/modules/detector.hh"
#include "blade/modules/cast.hh"
#include "blade/modules/polarizer.hh"

namespace Blade::Bundles::Generic {

template<typename IT, typename OT>
class BLADE_API ModeH : public Bundle {
 public:
    // Configuration 

    struct Config {
        ArrayShape inputShape;

        BOOL polarizerConvertToCircular = false;

        U64 detectorIntegrationSize;
        U64 detectorNumberOfOutputPolarizations;

        U64 castBlockSize = 512;
        U64 polarizerBlockSize = 512;
        U64 channelizerBlockSize = 512;
        U64 detectorBlockSize = 512;
    };

    // Input

    struct Input {
        ArrayTensor<Device::CUDA, IT> buffer;
    };

    // Output 

    constexpr const ArrayTensor<Device::CUDA, OT>& getOutputBuffer() {
        return detector->getOutputBuffer();
    }

    // Constructor

    explicit ModeH(const Config& config, const Input& input, const cudaStream_t& stream)
         : Bundle(stream), config(config) {
        BL_DEBUG("Initializing Pipeline Mode H.");

        if constexpr (!std::is_same<IT, CF32>::value) {
            BL_DEBUG("Instantiating input cast from {} to CF32.", TypeInfo<IT>::name);
            this->connect(cast, {
                .blockSize = config.castBlockSize,
            }, {
                .buf = input.buffer,
            });
        }

        BL_DEBUG("Instantiating channelizer with rate {}.", config.inputShape.numberOfTimeSamples());
        this->connect(channelizer, {
            .rate = config.inputShape.numberOfTimeSamples(),
            .blockSize = config.channelizerBlockSize,
        }, {
            .buf = this->getChannelizerInput(),
        });

        BL_DEBUG("Instatiating polarizer module.")
        this->connect(polarizer, {
            .mode = (config.polarizerConvertToCircular) ? Polarizer::Mode::XY2LR : Polarizer::Mode::BYPASS, 
            .blockSize = config.polarizerBlockSize,
        }, {
            .buf = channelizer->getOutputBuffer(),
        });

        BL_DEBUG("Instantiating detector module.");
        this->connect(detector, {
            .integrationSize = config.detectorIntegrationSize,
            .numberOfOutputPolarizations = config.detectorNumberOfOutputPolarizations,

            .blockSize = config.detectorBlockSize,
        }, {
            .buf = polarizer->getOutputBuffer(),
        });

        // TODO: Add output cast.
    }

 private:
    const Config config;

    using InputCast = typename Modules::Cast<CF16, CF32>;
    std::shared_ptr<InputCast> cast;

    using Channelizer = typename Modules::Channelizer<CF32, CF32>;
    std::shared_ptr<Channelizer> channelizer;

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

}  // namespace Blade::Bundles::Generic

#endif
