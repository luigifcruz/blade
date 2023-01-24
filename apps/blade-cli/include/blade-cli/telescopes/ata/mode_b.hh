#ifndef BLADE_CLI_TELESCOPES_ATA_MODE_B_HH
#define BLADE_CLI_TELESCOPES_ATA_MODE_B_HH

#include "blade-cli/types.hh"

#include "blade/plan.hh"
#include "blade/runner.hh"
#include "blade/utils/indicators.hh"
#include "blade/pipelines/ata/mode_b.hh"
#include "blade/pipelines/generic/mode_h.hh"
#include "blade/pipelines/generic/file_reader.hh"
#include "blade/pipelines/generic/accumulator.hh"

using namespace indicators;

namespace Blade::CLI::Telescopes::ATA {

template<typename IT, typename OT>
inline const Result ModeB(const Config& config) {
    // Define some types.
    using Reader = Pipelines::Generic::FileReader<IT>;
    using Channelize = Pipelines::Generic::ModeH<IT, CF32>;
    using Compute = Pipelines::ATA::ModeB<CF32, OT>;
    using GuppiWriter = Pipelines::Generic::Accumulator<Modules::Guppi::Writer<OT>, Device::CPU, OT>;
    std::unique_ptr<Runner<GuppiWriter>> guppiWriterRunner;
    using FilterbankWriter = Pipelines::Generic::Accumulator<Modules::Filterbank::Writer<OT>, Device::CPU, OT>;
    std::unique_ptr<Runner<FilterbankWriter>> filterbankWriterRunner;

    ArrayDimensions stepTailIncrementDims = {.A=1, .F=1, .T=1, .P=1};

    // Instantiate reader pipeline and runner.

    typename Reader::Config readerConfig = {
        .inputGuppiFile = config.inputGuppiFile,
        .inputBfr5File = config.inputBfr5File,
        .stepNumberOfTimeSamples = config.preBeamformerChannelizerRate,
        .stepNumberOfFrequencyChannels = config.stepNumberOfFrequencyChannels,
    };

    auto readerRunner = Runner<Reader>::New(1, readerConfig, false);
    const auto& reader = readerRunner->getWorker();

    const auto readerTotalOutputDims = reader.getTotalOutputDims();
    const auto readerStepOutputDims = reader.getStepOutputDims();
    const auto readerStepsInDims = readerTotalOutputDims/readerStepOutputDims;

    // Instantiate channelize pipeline and runner.

    typename Channelize::Config channelizeConfig = {
        .inputDimensions = readerStepOutputDims,

        .accumulateRate = 1,

        .polarizerConvertToCircular = false,

        .detectorIntegrationSize = 1,
        .detectorNumberOfOutputPolarizations = 2,

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

    // Instantiate compute pipeline and runner.

    // TODO: less static hardware limit `1024`
    const auto timeRelatedBlockSize = config.stepNumberOfTimeSamples > 1024 ? 1024 : config.stepNumberOfTimeSamples;

    const auto filterbankOutput = config.outputType == TypeId::F16 || config.outputType == TypeId::F32;

    // Device (CUDA) transposition is faster, but uses way too much CUDA memory.
    // TODO rectify this: seems the CUDA Graph takes up a lot of memory when the fine-channel count is high
    //  on account of the transposition being a `for(i : A*F)` loop of memcpy2D calls. Consider a kernel instead,
    //  or having the detector module output ATPF seeing as it is not inplace anyway.
    const auto deviceTransposition = filterbankOutput && true;

    typename Compute::Config computeConfig = {
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
        .phasorAntennaCoefficients = reader.getAntennaCoefficients(readerTotalOutputDims.numberOfFrequencyChannels(), reader.getChannelStartIndex()),
        .phasorBeamCoordinates = reader.getBeamCoordinates(),
        .phasorAntennaCoefficientChannelRate = config.preBeamformerChannelizerRate,

        .beamformerIncoherentBeam = false,

        .detectorEnable = filterbankOutput,
        .detectorIntegrationSize = 1,
        .detectorNumberOfOutputPolarizations = 1,
        .detectorTransposedATPFrevOutput = deviceTransposition,

        // TODO: Review this calculation.
        .castBlockSize = 32,
        .channelizerBlockSize = timeRelatedBlockSize,
        .phasorBlockSize = 32,
        .beamformerBlockSize = timeRelatedBlockSize,
        .detectorBlockSize = 32,
    };
    stepTailIncrementDims.T *= computeConfig.accumulateRate;
    if (readerStepsInDims.T < stepTailIncrementDims.T) {
        BL_FATAL("Reader cannot provide enough steps in Time for beamformer to gather {}!", computeConfig.accumulateRate);
        return Result::ASSERTION_ERROR;
    }
    if (readerStepsInDims.T % stepTailIncrementDims.T != 0) {
        BL_WARN("Reader does not provide a whole multiple of steps in Time for beamformer to gather {}.", computeConfig.accumulateRate);
    }
    BL_INFO("Beamformer gathers T={}.", computeConfig.accumulateRate);
    

    auto computeRunner = Runner<Compute>::New(config.numberOfWorkers, computeConfig, false);
    
    if constexpr (std::is_same<OT, CF32>::value || std::is_same<OT, CF16>::value) {
        typename GuppiWriter::Config writerConfig = {
            .moduleConfig = {
                .filepath = config.outputFile,
                .directio = true,
                .inputFrequencyBatches = 1, // Accumulator pipeline set to reconstituteBatchedDimensions.
            },
            .inputDimensions = computeRunner->getWorker().getOutputBuffer().dims(),
            .reconstituteBatchedDimensions = true,
            .accumulateRate = readerTotalOutputDims.numberOfFrequencyChannels() / readerStepOutputDims.numberOfFrequencyChannels(),
        };
        stepTailIncrementDims.F *= writerConfig.accumulateRate;
        if (readerStepsInDims.F < stepTailIncrementDims.F) {
            BL_FATAL("Reader cannot provide enough steps in Frequency for writer to gather {}!", writerConfig.accumulateRate);
            return Result::ASSERTION_ERROR;
        }
        if (readerStepsInDims.F % stepTailIncrementDims.F != 0) {
            BL_WARN("Reader does not provide a whole multiple of steps in Frequency for writer to gather {}.", writerConfig.accumulateRate);
        }
        BL_INFO("Writer gathers F={}.", writerConfig.accumulateRate);

        guppiWriterRunner = Runner<GuppiWriter>::New(1, writerConfig, false);
        auto& writer = guppiWriterRunner->getWorker();

        // Append information to the Accumulator's GUPPI header.

        writer.getModule()->headerPut("OBSFREQ", reader.getCenterFrequency()*1e-6);
        writer.getModule()->headerPut("OBSBW", reader.getBandwidth()*1e-6);
        writer.getModule()->headerPut("TBIN", config.preBeamformerChannelizerRate / reader.getChannelBandwidth());
        writer.getModule()->headerPut("PKTIDX", 0);
    }
    else if constexpr (std::is_same<OT, F32>::value || std::is_same<OT, F16>::value) {
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
                .numberOfIfChannels = (I32) computeRunner->getWorker().getOutputBuffer().dims().numberOfPolarizations(),
                .sourceName = reader.getSourceName(),
                .sourceDataFilename = config.inputGuppiFile,

                .numberOfInputFrequencyChannelBatches = 1, // Accumulator pipeline set to reconstituteBatchedDimensions.
            },
            .inputDimensions = computeRunner->getWorker().getOutputBuffer().dims(),
            .inputIsATPFNotAFTP = deviceTransposition, // ModeB.detectorTransposedATPFOutput is enabled
            .transposeATPF = ! deviceTransposition, // ModeB.detectorTransposedATPFOutput is enabled
            .reconstituteBatchedDimensions = true, 
            .accumulateRate = readerTotalOutputDims.numberOfFrequencyChannels() / readerStepOutputDims.numberOfFrequencyChannels(),
        };
        stepTailIncrementDims.F *= writerConfig.accumulateRate;
        if (readerStepsInDims.F < stepTailIncrementDims.F) {
            BL_FATAL("Reader cannot provide enough steps in Frequency for writer to gather {}!", writerConfig.accumulateRate);
            return Result::ASSERTION_ERROR;
        }
        if (readerStepsInDims.F % stepTailIncrementDims.F != 0) {
            BL_WARN("Reader does not provide a whole multiple of steps in Frequency for writer to gather {}.", writerConfig.accumulateRate);
        }
        BL_INFO("Writer gathers F={}.", writerConfig.accumulateRate);

        filterbankWriterRunner = Runner<FilterbankWriter>::New(1, writerConfig, false);
    }

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
    const U64 stepTailIncrement = stepTailIncrementDims.size();
    BL_DEBUG("Tail increments require {} steps ({}).", stepTailIncrement, stepTailIncrementDims);
    U64 callbackStep = 0;
    U64 workerId = 0;

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
            Plan::Available(computeRunner);

            // Try dequeue job from last runner. If unlucky, return.
            Plan::Dequeue(readerRunner, &callbackStep, &workerId);

            // Increment progress bar.
            bar.set_progress(static_cast<float>(stepCount) /
                    reader.getNumberOfSteps() * 100);

            // Compute input data.
            Plan::Compute(worker);

            auto& readerWorker = readerRunner->getWorker(workerId);
            // Concatenate output data inside search pipeline.
            Plan::Accumulate(computeRunner, channelizeRunner,
                             readerWorker.getStepOutputJulianDate(),
                             readerWorker.getStepOutputDut1(),
                             readerWorker.getStepOutputFrequencyChannelOffset(),
                             worker.getOutputBuffer());

            return callbackStep;
        });

        computeRunner->enqueue([&](auto& worker) {
            // Check if next runner has free slot.
            if constexpr (std::is_same<OT, CF32>::value || std::is_same<OT, CF16>::value) {
                Plan::Available(guppiWriterRunner);
            }
            else if constexpr (std::is_same<OT, F32>::value || std::is_same<OT, F16>::value) {
                Plan::Available(filterbankWriterRunner);
            }

            // Try dequeue job from last runner. If unlucky, return.
            Plan::Dequeue(channelizeRunner, &callbackStep);

            // Compute input data. 
            Plan::Compute(worker);

            // Concatenate output data inside writer pipeline.
            if constexpr (std::is_same<OT, CF32>::value || std::is_same<OT, CF16>::value) {
                Plan::Accumulate(guppiWriterRunner, computeRunner,
                                worker.getOutputBuffer());
            }
            else if constexpr (std::is_same<OT, F32>::value || std::is_same<OT, F16>::value) {
                Plan::Accumulate(filterbankWriterRunner, computeRunner,
                                worker.getOutputBuffer());
            }

            BL_DEBUG("Compute callbackStep: {}", callbackStep);
            return callbackStep;
        });

        if constexpr (std::is_same<OT, CF32>::value || std::is_same<OT, CF16>::value) {
            guppiWriterRunner->enqueue([&](auto& worker){
                // Try dequeue job from last runner. If unlucky, return.
                Plan::Dequeue(computeRunner, &callbackStep);

                // If accumulation complete, write data to disk.
                Plan::Compute(worker);

                BL_DEBUG("Writer callbackStep: {}", callbackStep);
                return callbackStep;
            });

            // Try to dequeue job from writer runner.
            if (guppiWriterRunner->dequeue(&callbackStep)) {
                BL_DEBUG("callbackStep: {}/{} ({})", callbackStep, reader.getNumberOfSteps(), stepTailIncrement);
                if (callbackStep + stepTailIncrement > reader.getNumberOfSteps()) {
                    break;
                }
            }    
        }
        else if constexpr (std::is_same<OT, F32>::value || std::is_same<OT, F16>::value) {
            filterbankWriterRunner->enqueue([&](auto& worker){
                // Try dequeue job from last runner. If unlucky, return.
                Plan::Dequeue(computeRunner, &callbackStep);

                // If accumulation complete, write data to disk.
                Plan::Compute(worker);

                BL_DEBUG("Writer callbackStep: {}", callbackStep);
                return callbackStep;
            });

            // Try to dequeue job from writer runner.
            if (filterbankWriterRunner->dequeue(&callbackStep)) {
                BL_DEBUG("callbackStep: {}/{} ({})", callbackStep, reader.getNumberOfSteps(), stepTailIncrement);
                if (callbackStep + stepTailIncrement > reader.getNumberOfSteps()) {
                    break;
                }
            }
        }
    }

    // Gracefully destroy runners. 

    readerRunner.reset();
    channelizeRunner.reset();
    computeRunner.reset();
    if constexpr (std::is_same<OT, CF32>::value || std::is_same<OT, CF16>::value) {
        guppiWriterRunner.reset();
    }
    else if constexpr (std::is_same<OT, F32>::value || std::is_same<OT, F16>::value) {
        filterbankWriterRunner.reset();
    }

    return Result::SUCCESS;
}

}  // namespace Blade::CLI::Telescopes::ATA

#endif
