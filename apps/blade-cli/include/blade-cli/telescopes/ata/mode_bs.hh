#ifndef BLADE_CLI_TELESCOPES_ATA_MODE_BS_HH
#define BLADE_CLI_TELESCOPES_ATA_MODE_BS_HH

#include "blade-cli/types.hh"
#include "blade-cli/telescopes/ata/config.hh"

#include "blade/plan.hh"
#include "blade/runner.hh"
#include "blade/utils/indicators.hh"
#include "blade/pipelines/generic/mode_h.hh"
#include "blade/pipelines/ata/mode_b.hh"
#include "blade/pipelines/generic/mode_s.hh"
#include "blade/pipelines/generic/file_reader.hh"
#include "blade/pipelines/generic/accumulator.hh"

using namespace indicators;

namespace Blade::CLI::Telescopes::ATA {

template<typename IT, typename OT>
inline const Result ModeBS(const Config& config) {
    // Define some types.
    using Reader = Pipelines::Generic::FileReader<IT>;
    using Channelize = Pipelines::Generic::ModeH<IT, CF32>;
    using Beamform = Pipelines::ATA::ModeB<CF32, F32>;
    using Search = Pipelines::Generic::ModeS<Blade::Pipelines::Generic::HitsFormat::SETICORE_STAMP>;

    ArrayDimensions stepTailIncrementDims = {.A=1, .F=1, .T=1, .P=1};

    // Instantiate reader pipeline and runner.
    // lock timesteps to channelizerRate, as channelization is fastest when there is only 1 output timestep
    //  (the config.stepNumberOfTimeSamples is accumulated for the Search input)

    typename Reader::Config readerConfig = {
        .inputGuppiFile = config.inputGuppiFile,
        .inputBfr5File = config.inputBfr5File,
        .stepNumberOfTimeSamples = config.preBeamformerChannelizerRate,
        .requiredMultipleOfTimeSamplesSteps = config.stepNumberOfTimeSamples,
        .stepNumberOfFrequencyChannels = config.stepNumberOfFrequencyChannels,
        .numberOfTimeSampleStepsBeforeFrequencyChannelStep = 0,
        .numberOfGuppiFilesLimit = config.inputGuppiFileLimit,
    };

    auto readerRunner = Runner<Reader>::New(1, readerConfig, false);
    const auto& reader = readerRunner->getWorker();
    
    const auto readerTotalOutputDims = reader.getTotalOutputDims();
    const auto readerStepsInDims = reader.getNumberOfStepsInDimensions();

    // Instantiate channelize pipeline and runner.

    typename Channelize::Config channelizeConfig = {
        .inputDimensions = reader.getStepOutputDims(),

        // will fully channelize each readerStepOutput (output.dims().T == 1)
        .accumulateRate = 1,

        .polarizerConvertToCircular = false,

        // Detector is disabled if OT != F32|F16
        .detectorIntegrationSize = 1,
        .detectorKernel = DetectorKernel::AFTP_4pol,

        .castBlockSize = 512,
        .polarizerBlockSize = 512,
        .channelizerBlockSize = 512,
        .detectorBlockSize = 512,
    };
    stepTailIncrementDims.T *= channelizeConfig.accumulateRate;
    if (readerStepsInDims.T < stepTailIncrementDims.T) {
        BL_FATAL("Reader cannot provide enough steps in Time for channelizer to gather {}!", channelizeConfig.accumulateRate);
        return Result::ASSERTION_ERROR;
    }
    if (readerStepsInDims.T % stepTailIncrementDims.T != 0) {
        BL_WARN("Reader does not provide a whole multiple of steps in Time for channelizer to gather {}.", channelizeConfig.accumulateRate);
    }
    BL_INFO("Channelizer gathers T={}.", channelizeConfig.accumulateRate);

    auto channelizeRunner = Runner<Channelize>::New(config.numberOfWorkers, channelizeConfig, false);

    // Instantiate beamform pipeline and runner.
    const auto timeRelatedBlockSize = config.stepNumberOfTimeSamples > 512 ? 512 : config.stepNumberOfTimeSamples;

    typename Beamform::Config beamformConfig = {
        .inputDimensions = channelizeRunner->getWorker().getOutputBuffer().dims(),
        .accumulateRate = config.stepNumberOfTimeSamples,

        .preBeamformerChannelizerRate = 1,

        .phasorBottomFrequencyHz = reader.getBottomFrequency(),
        .phasorChannelBandwidthHz = reader.getChannelBandwidth(),
        .phasorReferenceAntennaIndex = 0,
        .phasorArrayReferencePosition = reader.getReferencePosition(),
        .phasorBoresightCoordinate = reader.getPhaseCenterCoordinates(),
        .phasorAntennaPositions = reader.getAntennaPositions(),
        .phasorAntennaCoefficients = reader.getAntennaCoefficients(
            readerTotalOutputDims.numberOfFrequencyChannels(),
            reader.getChannelStartIndex()
        ),
        .phasorBeamCoordinates = reader.getBeamCoordinates(),
        .phasorAntennaCoefficientChannelRate = config.preBeamformerChannelizerRate,
        .phasorNegateDelays = config.phasorNegateDelays,

        .beamformerIncoherentBeam = true,

        .detectorEnable = true,
        .detectorIntegrationSize = config.integrationSize,
        .detectorKernel = DetectorKernel::ATPF_1pol,

        .castBlockSize = 512,
        .channelizerBlockSize = timeRelatedBlockSize,
        .phasorBlockSize = 512,
        .beamformerBlockSize = timeRelatedBlockSize,
        .detectorBlockSize = 512,
    };
    stepTailIncrementDims.T *= beamformConfig.accumulateRate;
    if (readerStepsInDims.T < stepTailIncrementDims.T) {
        BL_FATAL("Reader cannot provide enough steps in Time for beamformer to gather {}!", beamformConfig.accumulateRate);
        return Result::ASSERTION_ERROR;
    }
    if (readerStepsInDims.T % stepTailIncrementDims.T != 0) {
        BL_WARN("Reader does not provide a whole multiple of steps in Time for beamformer to gather {}.", beamformConfig.accumulateRate);
    }
    BL_INFO("Beamformer gathers T={}.", beamformConfig.accumulateRate);

    std::vector<std::string> beamSourceNames = reader.getBeamSourceNames();
    beamSourceNames.push_back("Incoherent");
    std::vector<RA_DEC> beamCoordinates = reader.getBeamCoordinates();
    beamCoordinates.push_back(reader.getPhaseCenterCoordinates());

    auto beamformRunner = Runner<Beamform>::New(config.numberOfWorkers, beamformConfig, false);

    // Instantiate search pipeline and runner.
    const auto stepNumberOfTimeSamples_prevPow2 = config.stepNumberOfTimeSamples == 0 ? 0 : (0x80000000 >> __builtin_clz(config.stepNumberOfTimeSamples));
    if (config.stepNumberOfTimeSamples != stepNumberOfTimeSamples_prevPow2) {
        BL_FATAL("Mode S requires a power of 2 number of spectra. Specify an appropriate number of step time-spectra.");
    }
    const auto searchAccumulateRate = readerTotalOutputDims.T / (config.preBeamformerChannelizerRate * config.stepNumberOfTimeSamples);
    const auto searchAccumulateRate_prevPow2 = searchAccumulateRate == 0 ? 0 : (0x80000000 >> __builtin_clz(searchAccumulateRate));
    if (searchAccumulateRate != searchAccumulateRate_prevPow2) {
        BL_FATAL("Mode S requires a power of 2 number of spectra. Accumulation rate limited from {} to {}.", searchAccumulateRate, searchAccumulateRate_prevPow2);
    }

    typename Search::Config searchConfig = {
        // accumulate all time
        .accumulateRate = searchAccumulateRate_prevPow2,

        .prebeamformerInputDimensions = beamformRunner->getWorker().getInputBuffer().dims(),
        .inputDimensions = beamformRunner->getWorker().getOutputBuffer().dims(),

        .inputTelescopeId = 0,
        .inputSourceName = reader.getSourceName(),
        .inputObservationIdentifier = "Unknown",
        .inputPhaseCenter = reader.getPhaseCenterCoordinates(),
        .inputTotalNumberOfTimeSamples = readerTotalOutputDims.numberOfFrequencyChannels() * config.preBeamformerChannelizerRate / config.integrationSize,
        .inputTotalNumberOfFrequencyChannels = readerTotalOutputDims.numberOfTimeSamples() / config.preBeamformerChannelizerRate,
        .inputFrequencyOfFirstChannelHz = reader.getBottomFrequency() + 0.5*reader.getChannelBandwidth()/config.preBeamformerChannelizerRate,
        .inputCoarseChannelRatio = config.preBeamformerChannelizerRate,
        .inputLastBeamIsIncoherent = true,
        .beamNames = beamSourceNames,
        .beamCoordinates = beamCoordinates,

        .searchMitigateDcSpike = true,
        .searchDriftRateZeroExcluded = config.driftRateZeroExcluded,
        .searchMinimumDriftRate = config.driftRateMinimum,
        .searchMaximumDriftRate = config.driftRateMaximum,
        .searchSnrThreshold = config.snrThreshold,
        .searchStampFrequencyMarginHz = config.stampFrequencyMarginHz,
        .searchHitsGroupingMargin = config.hitsGroupingMargin,
        .searchIncoherentBeam = true,

        .searchChannelBandwidthHz = reader.getChannelBandwidth() / config.preBeamformerChannelizerRate,
        .searchChannelTimespanS = config.preBeamformerChannelizerRate * config.integrationSize * reader.getChannelTimespan(),
        .searchOutputFilepathStem = config.outputFile,

        .produceDebugHits = config.produceDebugHits,
    };
    // searchConfig.accumulateRate;

    auto searchRunner = Runner<Search>::New(config.numberOfWorkers, searchConfig, false);

    indicators::ProgressBar bar{
        option::BarWidth{100},
        option::Start{" ["},
        option::Fill{"█"},
        option::Lead{"█"},
        option::Remainder{"-"},
        option::End{"]"},
        option::PrefixText{"Processing ATA::ModeBS"},
        option::ForegroundColor{Color::cyan},
        option::ShowElapsedTime{true},
        option::ShowRemainingTime{true},
        option::FontStyles{std::vector<FontStyle>{FontStyle::bold}}
    };

    // Run main processing loop.

    U64 stepCount = 0;
    const U64 stepSearchIncrement = stepTailIncrementDims.size();
    BL_DEBUG("Tail increments require {} steps ({}).", stepSearchIncrement, stepTailIncrementDims);

    U64 callbackStep = 0;
    U64 workerId = 0;

    const auto readerNumberOfSteps = readerStepsInDims.size(); // this accounts for numberOfTimeSampleStepsBeforeFrequencyChannelStep

    while (Plan::Loop()) {
        readerRunner->enqueue([&](auto& worker){
            // Check if next runner has free slot.
            Plan::Available(channelizeRunner);

            if (stepCount == readerNumberOfSteps) {
                BL_CHECK_THROW(Result::EXHAUSTED);
            }

            // Read block of data.
            Plan::Compute(worker);

            // Transfer output data from this pipeline to the next runner.
            Plan::Accumulate(channelizeRunner, readerRunner,
                            worker.getStepOutputBuffer());

            return ++stepCount;
        });

        channelizeRunner->enqueue([&](auto& worker) {
            // Check if next runner has free slot.
            Plan::Available(beamformRunner);

            // Try dequeue job from last runner. If unlucky, return.
            Plan::Dequeue(readerRunner, &callbackStep, &workerId);

            if (!config.progressBarDisabled) {
                // Increment progress bar.
                bar.set_progress(static_cast<float>(callbackStep) /
                        readerNumberOfSteps * 100);
            }

            // Compute input data.
            Plan::Compute(worker);

            auto& readerWorker = readerRunner->getWorker(workerId);
            // Concatenate output data inside beamform pipeline.
            Plan::Accumulate(beamformRunner, channelizeRunner,
                             readerWorker.getStepOutputJulianDate(),
                             readerWorker.getStepOutputDut1(),
                             readerWorker.getStepOutputFrequencyChannelOffset(),
                             worker.getOutputBuffer());

            return callbackStep;
        });

        beamformRunner->enqueue([&](auto& worker) {
            // Check if next runner has free slot.
            Plan::Available(searchRunner);

            // Try dequeue job from last runner. If unlucky, return.
            Plan::Dequeue(channelizeRunner, &callbackStep);

            // Compute input data.
            Plan::Compute(worker);

            // Concatenate output data inside search pipeline.
            Plan::Accumulate(
                searchRunner, beamformRunner,
                worker.getOutputBuffer(),
                worker.getInputBuffer(),
                worker.getBlockFrequencyChannelOffset(),
                worker.getBlockJulianDate()
            );

            return callbackStep;
        });

        searchRunner->enqueue([&](auto& worker){
            // Try dequeue job from last runner. If unlucky, return.
            Plan::Dequeue(beamformRunner, &callbackStep, &workerId);

            // If accumulation complete, dedoppler-search data.
            Plan::Compute(worker);

            return callbackStep;
        });

        // Try to dequeue job from the final runner.
        if (searchRunner->dequeue(&callbackStep, &workerId)) {
            if (callbackStep + stepSearchIncrement > readerNumberOfSteps) {
                break;
            }
        }
    }

    // Gracefully destroy runners.

    readerRunner.reset();
    channelizeRunner.reset();
    beamformRunner.reset();
    searchRunner.reset();

    return Result::SUCCESS;
}

}  // namespace Blade::CLI::Telescopes::ATA

#endif
