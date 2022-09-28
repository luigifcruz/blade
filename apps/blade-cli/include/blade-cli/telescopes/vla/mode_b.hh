#ifndef BLADE_CLI_TELESCOPES_VLA_MODE_HB_HH
#define BLADE_CLI_TELESCOPES_VLA_MODE_HB_HH

#include "blade-cli/types.hh"

#include "blade/plan.hh"
#include "blade/runner.hh"
#include "blade/utils/indicators.hh"
#include "blade/pipelines/vla/mode_b.hh"
#include "blade/modules/phasor/ata.hh"
#include "blade/pipelines/generic/file_reader.hh"

using namespace indicators;

namespace Blade::CLI::Telescopes::VLA {

template<typename IT, typename OT>
inline const Result ModeB(const Config& config) {
    // Define some types.
    using Reader = Pipelines::Generic::FileReader<IT>;
    using Compute = Pipelines::VLA::ModeB<IT, OT>;

    // Instantiate reader pipeline and runner.

    typename Reader::Config readerConfig = {
        .inputGuppiFile = config.inputGuppiFile,
        .inputBfr5File = config.inputBfr5File,
        .stepNumberOfTimeSamples = config.stepNumberOfTimeSamples * 
                                   config.preBeamformerChannelizerRate,
        .stepNumberOfFrequencyChannels = config.stepNumberOfFrequencyChannels,
    };

    auto readerRunner = Runner<Reader>::New(1, readerConfig, false);
    const auto& reader = readerRunner->getWorker();

    const auto readerStepOutputDims = reader.getStepOutputDims();
    const auto readerTotalOutputDims = reader.getTotalOutputDims();

    // Instantiate Phasor module for once off phasor generation

    Vector<Device::CPU, F64> julianDate({1});
    julianDate[0] = (1649366473.0 / 86400) + 2440587.5;
    Vector<Device::CPU, F64> dut1({1});
    dut1[0] = 0.0;

    auto phasorCoeffDims = reader.getStepOutputDims() * ArrayTensorDimensions({
        .A = 1,
        .F = config.preBeamformerChannelizerRate,
        .T = 0,
        .P = 1,
    });
    auto phasorAntennaCoeffs = ArrayTensor<Device::CPU, CF64>(reader.getAntennaCalibrationsDims(1));
    reader.fillAntennaCalibrations(1, phasorAntennaCoeffs);

    auto phasor = Modules::Phasor::ATA<CF32>(
        {
            .numberOfAntennas = phasorCoeffDims.numberOfAspects(),
            .numberOfFrequencyChannels = phasorCoeffDims.numberOfFrequencyChannels(),
            .numberOfPolarizations = phasorCoeffDims.numberOfPolarizations(),

            .observationFrequencyHz = reader.getObservationFrequency(),
            .channelBandwidthHz = reader.getChannelBandwidth(),
            .totalBandwidthHz = reader.getTotalBandwidth(),
            .frequencyStartIndex = reader.getChannelStartIndex(),

            .referenceAntennaIndex = 0,
            .arrayReferencePosition = reader.getReferencePosition(),
            .boresightCoordinate = reader.getBoresightCoordinate(),
            .antennaPositions = reader.getAntennaPositions(),
            .antennaCalibrations = ArrayTensor<Device::CPU, CF64>(phasorAntennaCoeffs),
            .beamCoordinates = reader.getBeamCoordinates(),

            .preBeamformerChannelizerRate = config.preBeamformerChannelizerRate,
        },
        {
            .blockJulianDate = julianDate,
            .blockDut1 = dut1,
        }
    );
    phasor.preprocess();

    // Instantiate compute pipeline and runner.

    // TODO: less static hardware limit `1024`
    const auto timeRelatedBlockSize = config.stepNumberOfTimeSamples > 1024 ? 1024 : config.stepNumberOfTimeSamples;

    typename Compute::Config computeConfig = {
        .inputDimensions = reader.getStepOutputDims(),

        .preBeamformerChannelizerRate = config.preBeamformerChannelizerRate,

        .beamformerPhasors = phasor.getOutputPhasors(),

        .beamformerNumberOfBeams = reader.getBeamCoordinates().size(),
        .beamformerIncoherentBeam = true,

        .detectorEnable = true,
        .detectorIntegrationSize = 1,
        .detectorNumberOfOutputPolarizations = 1,

        // TODO: Review this calculation.
        .castBlockSize = 32,
        .channelizerBlockSize = timeRelatedBlockSize,
        .beamformerBlockSize = timeRelatedBlockSize,
        .detectorBlockSize = 32,
    };

    auto computeRunner = Runner<Compute>::New(config.numberOfWorkers, computeConfig, false);

    indicators::ProgressBar bar{
        option::BarWidth{100},
        option::Start{" ["},
        option::Fill{"█"},
        option::Lead{"█"},
        option::Remainder{"-"},
        option::End{"]"},
        option::PrefixText{"Processing VLA::ModeB"},
        option::ForegroundColor{Color::cyan},
        option::ShowElapsedTime{true},
        option::ShowRemainingTime{true},
        option::FontStyles{std::vector<FontStyle>{FontStyle::bold}}
    };

    // Run main processing loop.

    U64 stepCount = 0;
    U64 callbackStep = 0;

    while (Plan::Loop()) {
        readerRunner->enqueue([&](auto& worker){
            // Check if next runner has free slot.
            Plan::Available(computeRunner); 

            // Read block of data.
            Plan::Compute(worker);

            // Transfer output data from this pipeline to the next runner.
            Plan::TransferOut(computeRunner, readerRunner,
                              worker.getStepOutputBuffer());

            return stepCount++;
        });

        computeRunner->enqueue([&](auto& worker) {
            // Try dequeue job from last runner. If unlucky, return.
            Plan::Dequeue(readerRunner, &callbackStep);

            // Increment progress bar.
            bar.set_progress(static_cast<float>(stepCount) / 
                    reader.getNumberOfSteps() * 100);

            // Compute input data. 
            Plan::Compute(worker);

            return callbackStep;
        });


        // Try to dequeue job from writer runner.
        if (computeRunner->dequeue(&callbackStep)) {
            if ((callbackStep + 1) == reader.getNumberOfSteps()) {
                break;
            }
        }    
    }

    // Gracefully destroy runners. 

    readerRunner.reset();
    computeRunner.reset();

    return Result::SUCCESS;
}

}  // namespace Blade::CLI::Telescopes::VLA

#endif
