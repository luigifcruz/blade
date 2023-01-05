#ifndef BLADE_CLI_TELESCOPES_ATA_MODE_B_HH
#define BLADE_CLI_TELESCOPES_ATA_MODE_B_HH

#include "blade-cli/types.hh"

#include "blade/plan.hh"
#include "blade/runner.hh"
#include "blade/utils/indicators.hh"
#include "blade/pipelines/ata/mode_b.hh"
#include "blade/pipelines/generic/file_reader.hh"
#include "blade/pipelines/generic/accumulator.hh"

using namespace indicators;

namespace Blade::CLI::Telescopes::ATA {

template<typename IT, typename OT>
inline const Result ModeB(const Config& config) {
    // Define some types.
    using Reader = Pipelines::Generic::FileReader<IT>;
    using Compute = Pipelines::ATA::ModeB<IT, OT>;
    using GuppiWriter = Pipelines::Generic::Accumulator<Modules::Guppi::Writer<OT>, Device::CPU, OT>;
    std::unique_ptr<Runner<GuppiWriter>> guppiWriterRunner;
    using FilterbankWriter = Pipelines::Generic::Accumulator<Modules::Filterbank::Writer<OT>, Device::CPU, OT>;
    std::unique_ptr<Runner<FilterbankWriter>> filterbankWriterRunner;

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

    const auto readerTotalOutputDims = reader.getTotalOutputDims();

    // Instantiate compute pipeline and runner.

    // TODO: less static hardware limit `1024`
    const auto timeRelatedBlockSize = config.stepNumberOfTimeSamples > 1024 ? 1024 : config.stepNumberOfTimeSamples;

    const auto filterbankOutput = config.outputType == TypeId::F16 || config.outputType == TypeId::F32;

    // Device (CUDA) transposition is faster, but uses way too much CUDA memory.
    // TODO rectify this: seems the CUDA Graph takes up a lot of memory when the fine-channel count is high
    //  on account of the transposition being a `for(i : A*F)` loop of memcpy2D calls. Consider a kernel instead,
    //  or having the detector module output ATPF seeing as it is not inplace anyway.
    const auto deviceTransposition = filterbankOutput && false;

    typename Compute::Config computeConfig = {
        .inputDimensions = reader.getStepOutputDims(),
        .preBeamformerChannelizerRate = config.preBeamformerChannelizerRate,

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
            .accumulateRate = readerTotalOutputDims.numberOfFrequencyChannels() / computeConfig.inputDimensions.numberOfFrequencyChannels(),
        };

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
            .accumulateRate = readerTotalOutputDims.numberOfFrequencyChannels() / computeConfig.inputDimensions.numberOfFrequencyChannels(),
        };

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
                              worker.getStepOutputFrequencyChannelOffset(),
                              worker.getStepOutputBuffer());

            return stepCount++;
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
            Plan::Dequeue(readerRunner, &callbackStep);

            // Increment progress bar.
            bar.set_progress(static_cast<float>(stepCount) / 
                    reader.getNumberOfSteps() * 100);

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

            return callbackStep;
        });

        if constexpr (std::is_same<OT, CF32>::value || std::is_same<OT, CF16>::value) {
            guppiWriterRunner->enqueue([&](auto& worker){
                // Try dequeue job from last runner. If unlucky, return.
                Plan::Dequeue(computeRunner, &callbackStep);

                // If accumulation complete, write data to disk.
                Plan::Compute(worker);

                return callbackStep;
            });

            // Try to dequeue job from writer runner.
            if (guppiWriterRunner->dequeue(&callbackStep)) {
                if ((callbackStep + 1) == reader.getNumberOfSteps()) {
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

                return callbackStep;
            });

            // Try to dequeue job from writer runner.
            if (filterbankWriterRunner->dequeue(&callbackStep)) {
                if ((callbackStep + 1) == reader.getNumberOfSteps()) {
                    break;
                }
            }
        }
    }

    // Gracefully destroy runners. 

    readerRunner.reset();
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
