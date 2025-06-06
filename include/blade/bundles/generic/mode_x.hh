#ifndef BLADE_BUNDLES_GENERIC_MODE_X_HH
#define BLADE_BUNDLES_GENERIC_MODE_X_HH

#include <vector>

#include "blade/bundle.hh"

#include "blade/modules/stacker.hh"
#include "blade/modules/caster.hh"
#include "blade/modules/channelizer/base.hh"
#include "blade/modules/correlator.hh"
#include "blade/modules/integrator.hh"

namespace Blade::Bundles::Generic {

template<typename IT, typename OT>
class BLADE_API ModeX : public Bundle {
 public:
    // Configuration

    struct Config {
        ArrayShape inputShape;
        ArrayShape outputShape;

        U64 preChannelizerStackerMultiplier = 1;

        bool channelizerBypass = false;

        U64 preCorrelatorStackerMultiplier = 1;

        U64 correlatorIntegrationRate = 1;
        U64 correlatorConjugateAntennaIndex = 1;
        bool correlatorUseSharedMemory = false;
        CALC_MODE correlatorCalculationMode = CALC_MODE::DOUBLE_PRECISION_FP;

        U64 postCorrelatorFrequencyIntegrationRate = 1;

        U64 stackerBlockSize = 512;
        U64 casterBlockSize = 512;
        U64 channelizerBlockSize = 512;
        U64 correlatorBlockSize = 32;
    };

    constexpr const Config& getConfig() const {
        return this->config;
    }

    // Input

    struct Input {
        const ArrayTensor<Device::CUDA, IT>& buffer;
    };

    constexpr const ArrayTensor<Device::CUDA, IT>& getInputBuffer() const {
        return this->input.buffer;
    }

    // Output

    constexpr const ArrayTensor<Device::CUDA, OT>& getOutputBuffer() {
        if (config.channelizerBypass) {
            return bypassCorrelator->getOutputBuffer();
        } else {
            return correlator->getOutputBuffer();
        }
    }

    // Constructor

    explicit ModeX(const Config& config, const Input& input, const Stream& stream)
         : Bundle(stream), config(config), input(input) {
        BL_DEBUG("Initializing Mode-X Bundle.");

        if (config.channelizerBypass) {
            BL_DEBUG("Instantiating pre-correlator stacker module.");
            this->connect(bypassPreCorrelatorStacker, {
                .axis = 2,
                .multiplier = config.preCorrelatorStackerMultiplier,

                .blockSize = config.stackerBlockSize,
            }, {
                .buf = input.buffer,
            });

            BL_DEBUG("Instantiating correlator module.");
            this->connect(bypassCorrelator, {
                .integrationRate = config.correlatorIntegrationRate,
                .conjugateAntennaIndex = config.correlatorConjugateAntennaIndex,
                .useSharedMemory = config.correlatorUseSharedMemory,
                .calculationMode = config.correlatorCalculationMode,

                .blockSize = config.correlatorBlockSize,
            }, {
                .buf = bypassPreCorrelatorStacker->getOutputBuffer(),
            });
        } else {
            BL_DEBUG("Instantiating pre-channelizer stacker module.");
            this->connect(preChannelizerStacker, {
                .axis = 2,
                .multiplier = config.preChannelizerStackerMultiplier,

                .blockSize = config.stackerBlockSize,
            }, {
                .buf = input.buffer,
            });

            BL_DEBUG("Instantiating input caster module.");
            this->connect(inputCaster, {
                .blockSize = config.casterBlockSize,
            }, {
                .buf = preChannelizerStacker->getOutputBuffer(),
            });

            BL_DEBUG("Instantiating channelizer module.");
            this->connect(channelizer, {
                .rate = config.inputShape.numberOfTimeSamples() *
                        config.preChannelizerStackerMultiplier,

                .blockSize = config.channelizerBlockSize,
            }, {
                .buf = inputCaster->getOutputBuffer(),
            });

            BL_DEBUG("Instantiating pre-correlator stacker module.");
            this->connect(preCorrelatorStacker, {
                .axis = 2,
                .multiplier = config.preCorrelatorStackerMultiplier,

                .blockSize = config.stackerBlockSize,
            }, {
                .buf = channelizer->getOutputBuffer(),
            });

            BL_DEBUG("Instantiating correlator module.");
            this->connect(correlator, {
                .integrationRate = config.correlatorIntegrationRate,
                .conjugateAntennaIndex = config.correlatorConjugateAntennaIndex,
                .useSharedMemory = config.correlatorUseSharedMemory,
                .calculationMode = config.correlatorCalculationMode,

                .blockSize = config.correlatorBlockSize,
            }, {
                .buf = preCorrelatorStacker->getOutputBuffer(),
            });
        }

        BL_DEBUG("Instantiating post-correlator frequency-integration module.");
        this->connect(postCorrelatorFrequencyIntegrator, {
            .size = config.postCorrelatorFrequencyIntegrationRate,
            .axis = 1, // F

            .blockSize = config.stackerBlockSize,
        }, {
            .buf = channelizer->getOutputBuffer(),
        });

        if (getOutputBuffer().shape() != config.outputShape) {
            BL_FATAL("Expected output buffer size ({}) mismatch with actual size ({}).",
                     config.outputShape, getOutputBuffer().shape());
            BL_CHECK_THROW(Result::ERROR);
        }
    }

 private:
    const Config config;
    Input input;

    using PreChannelizerStacker = typename Modules::Stacker<IT, IT>;
    std::shared_ptr<PreChannelizerStacker> preChannelizerStacker;

    using InputCaster = typename Modules::Caster<IT, CF32>;
    std::shared_ptr<InputCaster> inputCaster;

    using PreChannelizer = typename Modules::Channelizer<CF32, CF32>;
    std::shared_ptr<PreChannelizer> channelizer;

    using PreCorrelatorStacker = typename Modules::Stacker<CF32, CF32>;
    std::shared_ptr<PreCorrelatorStacker> preCorrelatorStacker;

    using Correlator = typename Modules::Correlator<CF32, CF32>;
    std::shared_ptr<Correlator> correlator;

    using BypassPreCorrelatorStacker = typename Modules::Stacker<IT, IT>;
    std::shared_ptr<BypassPreCorrelatorStacker> bypassPreCorrelatorStacker;

    using BypassCorrelator = typename Modules::Correlator<IT, CF32>;
    std::shared_ptr<BypassCorrelator> bypassCorrelator;

    using CorrelatorIntegrator = typename Modules::Integrator<CF32, CF32>;
    std::shared_ptr<CorrelatorIntegrator> postCorrelatorFrequencyIntegrator;
};

}  // namespace Blade::Bundles::Generic

#endif
