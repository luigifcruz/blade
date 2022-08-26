#ifndef BLADE_CLI_TELESCOPES_ATA
#define BLADE_CLI_TELESCOPES_ATA

#include "types.hh"

#include "blade/plan.hh"
#include "blade/runner.hh"
#include "blade/pipelines/generic/file_writer.hh"
#include "blade/utils/indicators.hh"

#ifdef BLADE_PIPELINE_ATA_MODE_B
#include "blade/pipelines/ata/mode_b.hh"
#endif

using namespace indicators;

template<typename IT, typename OT>
inline const Result SetupAtaModeB(const CliConfig& cliConfig, 
                                  auto& readerRunner) {
    // Define some types.

    using Compute = Pipelines::ATA::ModeB<OT>;
    using Writer = Pipelines::Generic::FileWriter<OT>;

    // Get reader pipeline from runner to make things tidier.

    const auto& reader = readerRunner->getWorker();

    // Instantiate compute pipeline and runner.

    typename Compute::Config computeConfig = {
        .preBeamformerChannelizerRate = cliConfig.preBeamformerChannelizerRate,

        .phasorObservationFrequencyHz = reader.getObservationFrequency(),
        .phasorChannelBandwidthHz = reader.getChannelBandwidth(),
        .phasorTotalBandwidthHz = reader.getTotalBandwidth(),
        .phasorFrequencyStartIndex = reader.getChannelStartIndex(),
        .phasorReferenceAntennaIndex = 0,
        .phasorArrayReferencePosition = reader.getReferencePosition(),
        .phasorBoresightCoordinate = reader.getBoresightCoordinate(),
        .phasorAntennaPositions = reader.getAntennaPositions(),
        .phasorAntennaCalibrations = reader.getAntennaCalibrations(cliConfig.preBeamformerChannelizerRate),
        .phasorBeamCoordinates = reader.getBeamCoordinates(),

        .beamformerNumberOfAntennas = reader.getStepNumberOfAntennas(),
        .beamformerNumberOfFrequencyChannels = reader.getStepNumberOfFrequencyChannels(),
        .beamformerNumberOfTimeSamples = reader.getStepNumberOfTimeSamples(),
        .beamformerNumberOfPolarizations = reader.getStepNumberOfPolarizations(),
        .beamformerNumberOfBeams = reader.getStepNumberOfBeams(),
        .beamformerIncoherentBeam = false,

        .castBlockSize = 32,
        .channelizerBlockSize = cliConfig.stepNumberOfTimeSamples,
        .phasorBlockSize = 32,
        .beamformerBlockSize = cliConfig.stepNumberOfTimeSamples
    };

    auto computeRunner = Runner<Compute>::New(cliConfig.numberOfWorkers, computeConfig, false);

    // Instantiate writer pipeline and runner.

    typename Writer::Config writerConfig = {
        .outputGuppiFile = cliConfig.outputGuppiFile,
        .directio = true,

        .stepNumberOfBeams = reader.getStepNumberOfBeams(),
        .stepNumberOfAntennas = reader.getStepNumberOfAntennas(),
        .stepNumberOfFrequencyChannels = cliConfig.preBeamformerChannelizerRate * reader.getStepNumberOfFrequencyChannels(),
        .stepNumberOfTimeSamples = cliConfig.stepNumberOfTimeSamples,
        .stepNumberOfPolarizations = reader.getStepNumberOfPolarizations(),

        .totalNumberOfFrequencyChannels = cliConfig.preBeamformerChannelizerRate * reader.getTotalNumberOfFrequencyChannels(),
    };

    auto writerRunner = Runner<Writer>::New(1, writerConfig, false);
    auto& writer = writerRunner->getWorker();

    // Append information to the FileWriter's GUPPI header.

    writer.headerPut("OBSFREQ", reader.getObservationFrequency());
    writer.headerPut("OBSBW", reader.getChannelBandwidth() * 
                              writer.getTotalNumberOfFrequencyChannels() * 
                              writer.getStepNumberOfAntennas() * 
                              writer.getStepNumberOfBeams());
    writer.headerPut("TBIN", cliConfig.preBeamformerChannelizerRate / reader.getChannelBandwidth());
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
                             worker.getOutput());

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

template<typename IT, typename OT>
inline const Result SetupAta(const CliConfig& config,
                             auto& readerRunner) {
    switch (config.mode) {
#if defined(BLADE_PIPELINE_ATA_MODE_B)
        case ModeId::MODE_B:
            return SetupAtaModeB<IT, OT>(config, readerRunner);
#endif
#if defined(BLADE_PIPELINE_ATA_MODE_B) && defined(BLADE_PIPELINE_GENERIC_MODE_H)
#endif
        default:
            BL_FATAL("This ATA mode is not implemented yet.");
    }

    return Result::ERROR;
}

#endif
