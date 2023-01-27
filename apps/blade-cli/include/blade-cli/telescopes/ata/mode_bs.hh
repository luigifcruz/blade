#ifndef BLADE_CLI_TELESCOPES_ATA_MODE_BS_HH
#define BLADE_CLI_TELESCOPES_ATA_MODE_BS_HH

#include "blade-cli/types.hh"

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

template<typename IT>
inline const Result ModeBS(const Config& config) {
    // Define some types.
    using Reader = Pipelines::Generic::FileReader<IT>;
    using Channelize = Pipelines::Generic::ModeH<IT, CF32>;
    using Beamform = Pipelines::ATA::ModeB<CF32, F32>;
    using Search = Pipelines::Generic::ModeS<Blade::Pipelines::Generic::HitsFormat::SETICORE_STAMP>;
    using FilterbankWriter = Pipelines::Generic::Accumulator<Modules::Filterbank::Writer<F32>, Device::CPU, F32>;
    std::unique_ptr<Runner<FilterbankWriter>> filterbankWriterRunner;

    ArrayDimensions stepTailIncrementDims = {.A=1, .F=1, .T=1, .P=1};

    // Instantiate reader pipeline and runner.
    // lock timesteps to channelizerRate, as channelization is fastest when there is only 1 output timestep
    //  (the config.stepNumberOfTimeSamples is accumulated for the Search input)

    typename Reader::Config readerConfig = {
        .inputGuppiFile = config.inputGuppiFile,
        .inputBfr5File = config.inputBfr5File,
        .stepNumberOfTimeSamples = config.preBeamformerChannelizerRate,
        .stepNumberOfFrequencyChannels = config.stepNumberOfFrequencyChannels,
        .stepTimeSamplesFirstNotFrequencyChannels = true,
    };

    auto readerRunner = Runner<Reader>::New(1, readerConfig, false);
    const auto& reader = readerRunner->getWorker();

    
    const auto readerTotalOutputDims = reader.getTotalOutputDims();
    const auto readerStepOutputDims = reader.getStepOutputDims();
    const auto readerStepsInDims = readerTotalOutputDims/readerStepOutputDims;

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

    typename Beamform::Config beamformConfig = {
        .inputDimensions = channelizeRunner->getWorker().getOutputBuffer().dims(),
        .accumulateRate = config.stepNumberOfTimeSamples,

        .preBeamformerChannelizerRate = 1,

        .phasorObservationFrequencyHz = reader.getObservationFrequency(),
        .phasorChannelBandwidthHz = reader.getChannelBandwidth(),
        .phasorTotalBandwidthHz = reader.getObservationBandwidth(),
        .phasorFrequencyStartIndex = reader.getChannelStartIndex(),
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

        .beamformerIncoherentBeam = true,

        .detectorEnable = true,
        .detectorIntegrationSize = 1,
        .detectorKernel = DetectorKernel::ATPF_1pol,

        // TODO: Review this calculation.
        .castBlockSize = 512,
        .channelizerBlockSize = 512,
        .phasorBlockSize = 512,
        .beamformerBlockSize = config.stepNumberOfTimeSamples,
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

    auto beamformRunner = Runner<Beamform>::New(config.numberOfWorkers, beamformConfig, false);

    // Instantiate search pipeline and runner.

    typename Search::Config searchConfig = {
        .prebeamformerInputDimensions = beamformRunner->getWorker().getInputBuffer().dims(),
        .inputDimensions = beamformRunner->getWorker().getOutputBuffer().dims(),

        .inputTelescopeId = 0,
        .inputSourceName = reader.getSourceName(),
        .inputObservationIdentifier = "Unknown",
        .inputPhaseCenter = reader.getPhaseCenterCoordinates(),
        .inputTotalNumberOfTimeSamples = readerTotalOutputDims.numberOfFrequencyChannels() * config.preBeamformerChannelizerRate,
        .inputTotalNumberOfFrequencyChannels = readerTotalOutputDims.numberOfTimeSamples() / config.preBeamformerChannelizerRate,
        .inputCoarseStartChannelIndex = reader.getChannelStartIndex(),
        .inputJulianDateStart = reader.getJulianDateOfLastReadBlock(),
        .inputCoarseChannelRatio = config.preBeamformerChannelizerRate,
        .inputLastBeamIsIncoherent = true,
        .beamNames = reader.getBeamSourceNames(),
        .beamCoordinates = reader.getBeamCoordinates(),

        .searchMitigateDcSpike = true,
        .searchMinimumDriftRate = 0.0,
        .searchMaximumDriftRate = 10.0,
        .searchSnrThreshold = 30.0,

        .searchChannelBandwidthHz = reader.getChannelBandwidth() / config.preBeamformerChannelizerRate,
        .searchChannelTimespanS = config.preBeamformerChannelizerRate / reader.getChannelBandwidth(),
        .searchOutputFilepathStem = config.outputFile,
    };
    // searchConfig.accumulateRate;

    auto searchRunner = Runner<Search>::New(config.numberOfWorkers, searchConfig, false);

    // Instantiate writer pipeline and runner.
    
    typename FilterbankWriter::Config writerConfig = {
        .moduleConfig = {
            .filepath = config.outputFile,
            
            .machineId = 0,
            .telescopeName = reader.getTelescopeName(),
            .baryCentric = 1,
            .pulsarCentric = 1,
            .sourceCoordinate = reader.getPhaseCenterCoordinates(),
            .azimuthStart = reader.getAzimuthAngle(),
            .zenithStart = reader.getZenithAngle(),
            .centerFrequencyHz = reader.getCenterFrequency(),
            .bandwidthHz = -1*reader.getBandwidth(), // Negated as frequencies are reversed
            .julianDateStart = reader.getJulianDateOfLastReadBlock(),
            .numberOfIfChannels = (I32) beamformRunner->getWorker().getOutputBuffer().dims().numberOfPolarizations(),
            .sourceName = reader.getSourceName(),
            .sourceDataFilename = config.inputGuppiFile,

            .numberOfInputFrequencyChannelBatches = 1, // Accumulator pipeline set to reconstituteBatchedDimensions.
        },
        .inputDimensions = beamformRunner->getWorker().getOutputBuffer().dims(),
        .inputIsATPFNotAFTP = true, // ModeB.detectorTransposedATPFOutput is enabled
        .reconstituteBatchedDimensions = true,
        .accumulateRate = readerTotalOutputDims.numberOfFrequencyChannels() / reader.getStepOutputDims().numberOfFrequencyChannels(),
    };

    const auto readerTotalTimeSteps = readerTotalOutputDims.numberOfTimeSamples() / reader.getStepOutputDims().numberOfTimeSamples();
    const auto filterbankOutputEnabled = readerTotalTimeSteps == config.stepNumberOfTimeSamples;
    if (filterbankOutputEnabled) {
        BL_INFO(
            "The beamformer steps all time-samples at a time (-T = {} == {}), so the filterbank output is enabled.",
            config.stepNumberOfTimeSamples, readerTotalTimeSteps
        );
        // stepTailIncrementDims.F *= writerConfig.accumulateRate;
        // if (readerStepsInDims.F < stepTailIncrementDims.F) {
        //     BL_FATAL("Reader cannot provide enough steps in Frequency for writer to gather {}!", writerConfig.accumulateRate);
        //     return Result::ASSERTION_ERROR;
        // }
        // if (readerStepsInDims.F % stepTailIncrementDims.F != 0) {
        //     BL_WARN("Reader does not provide a whole multiple of steps in Frequency for writer to gather {}.", writerConfig.accumulateRate);
        // }
        // BL_INFO("Writer gathers F={}.", writerConfig.accumulateRate);

        filterbankWriterRunner = Runner<FilterbankWriter>::New(1, writerConfig, false);
    } else {
        BL_INFO(
            "The beamformer does not step all time-samples at a time (-T = {} != {}), so the filterbank output will be disabled.",
            config.stepNumberOfTimeSamples, readerTotalTimeSteps
        );
    }

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

    std::unordered_map<U64, Vector<Device::CPU, F64>> stepJulianDateMap;
    std::unordered_map<U64, Vector<Device::CPU, F64>> stepDut1Map;
    std::unordered_map<U64, Vector<Device::CPU, U64>> stepFrequencyChannelOffsetMap;

    stepJulianDateMap.reserve(config.numberOfWorkers);
    stepDut1Map.reserve(config.numberOfWorkers);
    stepFrequencyChannelOffsetMap.reserve(config.numberOfWorkers);

    U64 stepCount = 0;
    const U64 stepSearchIncrement = stepTailIncrementDims.size();
    BL_DEBUG("Tail increments require {} steps ({}).", stepSearchIncrement, stepTailIncrementDims);
    stepTailIncrementDims.F *= writerConfig.accumulateRate;
    const U64 stepFilterbankIncrement = stepTailIncrementDims.size();
    U64 callbackStep = 0;
    U64 workerId = 0;

    BOOL writeComplete = !filterbankOutputEnabled;
    BOOL searchComplete = false;

    while (Plan::Loop()) {
        readerRunner->enqueue([&](auto& worker){
            // Check if next runner has free slot.
            Plan::Available(channelizeRunner);

            if (stepCount == reader.getNumberOfSteps()) {
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

            // Increment progress bar.
            bar.set_progress(static_cast<float>(callbackStep) /
                    reader.getNumberOfSteps() * 100);

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
            Plan::Accumulate(searchRunner, beamformRunner,
                             worker.getOutputBuffer(),
                             worker.getInputBuffer(),
                             worker.getBlockFrequencyChannelOffset());
            if (filterbankOutputEnabled) {
                Plan::Accumulate(filterbankWriterRunner, beamformRunner,
                                worker.getOutputBuffer());
            }

            return callbackStep;
        });

        searchRunner->enqueue([&](auto& worker){
            // Try dequeue job from last runner. If unlucky, return.
            Plan::Dequeue(beamformRunner, &callbackStep, &workerId);
            
            auto& beamformWorker = beamformRunner->getWorker(workerId);
            
            stepJulianDateMap.insert({callbackStep, beamformWorker.getBlockJulianDate()});
            stepDut1Map.insert({callbackStep, beamformWorker.getBlockDut1()});
            stepFrequencyChannelOffsetMap.insert({callbackStep, beamformWorker.getBlockFrequencyChannelOffset()});

            if (filterbankOutputEnabled) {
                // write out the input to the searchRunner
                filterbankWriterRunner->enqueue([&](auto& worker){
                    // If accumulation complete, write data to disk.
                    Plan::Compute(worker);

                    return callbackStep;
                });
            }

            worker.setFrequencyOfFirstInputChannel(
                reader.getCenterFrequency()
                + (
                    beamformWorker.getBlockFrequencyChannelOffset()[0]
                    - readerTotalOutputDims.numberOfFrequencyChannels() / 2
                ) * reader.getChannelBandwidth()
            );

            // If accumulation complete, dedoppler-search data.
            Plan::Compute(worker);

            return callbackStep;
        });

        // Try to dequeue job from the final runner.
        if (searchRunner->dequeue(&callbackStep, &workerId)) {
            if (callbackStep + stepSearchIncrement > reader.getNumberOfSteps()) {
                searchComplete = true;
            }
        }

        if (filterbankOutputEnabled) {
            // Try to dequeue job from the writer runner.
            if (filterbankWriterRunner->dequeue(&callbackStep, &workerId)) {
                BL_INFO("{}: Filterbank written, containing the input to the dedoppler search.", callbackStep);
                stepJulianDateMap.erase(callbackStep);
                stepDut1Map.erase(callbackStep);
                stepFrequencyChannelOffsetMap.erase(callbackStep);

                if (callbackStep + stepFilterbankIncrement > reader.getNumberOfSteps()) {
                    writeComplete = true;
                }
            }
        }

        if (writeComplete && searchComplete) {
            break;
        }
    }

    // Gracefully destroy runners.

    readerRunner.reset();
    channelizeRunner.reset();
    beamformRunner.reset();
    searchRunner.reset();
    filterbankWriterRunner.reset();

    return Result::SUCCESS;
}

}  // namespace Blade::CLI::Telescopes::ATA

#endif
