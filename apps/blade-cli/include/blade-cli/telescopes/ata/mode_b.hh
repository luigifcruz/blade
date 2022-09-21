#ifndef BLADE_CLI_TELESCOPES_ATA_MODE_B_HH
#define BLADE_CLI_TELESCOPES_ATA_MODE_B_HH

#include "blade-cli/types.hh"

#include "blade/plan.hh"
#include "blade/runner.hh"
#include "blade/utils/indicators.hh"
#include "blade/pipelines/ata/mode_b.hh"
#include "blade/pipelines/generic/file_reader.hh"
#include "blade/pipelines/generic/file_writer.hh"

using namespace indicators;

namespace Blade::CLI::Telescopes::ATA {

template<typename IT, typename OT>
inline const Result ModeB(const Config& config) {
    // Define some types.
    using Reader = Pipelines::Generic::FileReader<IT>;
    using Compute = Pipelines::ATA::ModeB<OT>;
    using Writer = Pipelines::Generic::FileWriter<OT>;

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

    // Instantiate compute pipeline and runner.

    typename Compute::Config computeConfig = {
        .inputDimensions = {
            .A = reader.getStepNumberOfAntennas(),
            .F = reader.getStepNumberOfFrequencyChannels(),
            .T = reader.getStepNumberOfTimeSamples(),
            .P = reader.getStepNumberOfPolarizations(),
        },
        .preBeamformerChannelizerRate = config.preBeamformerChannelizerRate,

        .phasorObservationFrequencyHz = reader.getObservationFrequency(),
        .phasorChannelBandwidthHz = reader.getChannelBandwidth(),
        .phasorTotalBandwidthHz = reader.getTotalBandwidth(),
        .phasorFrequencyStartIndex = reader.getChannelStartIndex(),
        .phasorReferenceAntennaIndex = 0,
        .phasorArrayReferencePosition = reader.getReferencePosition(),
        .phasorBoresightCoordinate = reader.getBoresightCoordinate(),
        .phasorAntennaPositions = reader.getAntennaPositions(),
        .phasorAntennaCalibrations = {},
        .phasorBeamCoordinates = reader.getBeamCoordinates(),

        .beamformerIncoherentBeam = false,

        .detectorEnable = false,
        // .detectorIntegrationSize,
        // .detectorNumberOfOutputPolarizations,

        // TODO: Review this calculation.
        .castBlockSize = 32,
        .channelizerBlockSize = config.stepNumberOfTimeSamples,
        .phasorBlockSize = 32,
        .beamformerBlockSize = config.stepNumberOfTimeSamples,
        .detectorBlockSize = 32,
    };

    computeConfig.phasorAntennaCalibrations.resize(reader.getAntennaCalibrationsDims(config.preBeamformerChannelizerRate));
    reader.fillAntennaCalibrations(config.preBeamformerChannelizerRate, computeConfig.phasorAntennaCalibrations);

    auto computeRunner = Runner<Compute>::New(config.numberOfWorkers, computeConfig, false);

    // Instantiate writer pipeline and runner.

    typename Writer::Config writerConfig = {
        .outputGuppiFile = config.outputGuppiFile,
        .directio = true,
        .inputDimensions = computeRunner->getWorker().getOutputBuffer().dims(),
        .accumulateRate = reader.getTotalNumberOfFrequencyChannels() / reader.getStepNumberOfFrequencyChannels()
    };

    auto writerRunner = Runner<Writer>::New(1, writerConfig, false);
    auto& writer = writerRunner->getWorker();

    // Append information to the FileWriter's GUPPI header.

    writer.headerPut("OBSFREQ", reader.getObservationFrequency());
    writer.headerPut("OBSBW", reader.getChannelBandwidth() * 
                              reader.getTotalNumberOfFrequencyChannels());
    writer.headerPut("TBIN", config.preBeamformerChannelizerRate / reader.getChannelBandwidth());
    writer.headerPut("PKTIDX", 0);

    indicators::ProgressBar bar{
        option::BarWidth{100},
        option::Start{" ["},
        option::Fill{"█"},
        option::Lead{"█"},
        option::Remainder{"-"},
        option::End{"]"},
        option::PrefixText{"Processing ATA::ModeB"},
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
                              worker.getStepOutputJulianDate(),
                              worker.getStepOutputDut1(),
                              worker.getStepOutputBuffer());

            return stepCount++;
        });

        computeRunner->enqueue([&](auto& worker) {
            // Check if next runner has free slot.
            Plan::Available(writerRunner);

            // Try dequeue job from last runner. If unlucky, return.
            Plan::Dequeue(readerRunner, &callbackStep);

            // Increment progress bar.
            bar.set_progress(static_cast<float>(stepCount) / 
                    reader.getNumberOfSteps() * 100);

            // Compute input data. 
            Plan::Compute(worker);

            // Concatenate output data inside writer pipeline.
            Plan::Accumulate(writerRunner, computeRunner,
                             worker.getOutputBuffer());

            return callbackStep;
        });

        writerRunner->enqueue([&](auto& worker){
            // Try dequeue job from last runner. If unlucky, return.
            Plan::Dequeue(computeRunner, &callbackStep);

            // If accumulation complete, write data to disk.
            Plan::Compute(worker);

            return callbackStep;
        });

        // Try to dequeue job from writer runner.
        if (writerRunner->dequeue(&callbackStep)) {
            if ((callbackStep + 1) == reader.getNumberOfSteps()) {
                break;
            }
        }    
    }

    // Gracefully destroy runners. 

    readerRunner.reset();
    computeRunner.reset();
    writerRunner.reset();

    return Result::SUCCESS;
}

}  // namespace Blade::CLI::Telescopes::ATA

#endif
