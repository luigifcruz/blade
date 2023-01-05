#ifndef BLADE_CLI_TELESCOPES_ATA_MODE_BS_HH
#define BLADE_CLI_TELESCOPES_ATA_MODE_BS_HH

#include "blade-cli/types.hh"

#include "blade/plan.hh"
#include "blade/runner.hh"
#include "blade/utils/indicators.hh"
#include "blade/pipelines/generic/mode_h.hh"
#include "blade/pipelines/ata/mode_b.hh"
#include "blade/pipelines/generic/mode_s.hh"

using namespace indicators;

namespace Blade::CLI::Telescopes::ATA {

template<typename IT>
inline const Result ModeBS(const Config& config) {
    // Define some types.
    using Reader = Pipelines::Generic::FileReader<IT>;
    using Channelize = Pipelines::Generic::ModeH<IT, CF32>;
    using Beamform = Pipelines::ATA::ModeB<CF32, F32>;
    using Search = Pipelines::Generic::ModeS;

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

    // Instantiate channelize pipeline and runner.

    typename Channelize::Config channelizeConfig = {
        .inputDimensions = reader.getStepOutputDims(),

        .accumulateRate = 1, // will fully channelize each readerStepOutput (output.dims().T == 1)

        .polarizerConvertToCircular = false,

        .detectorIntegrationSize = 1,
        .detectorNumberOfOutputPolarizations = 1,

        .castBlockSize = 512,
        .polarizerBlockSize = 512,
        .channelizerBlockSize = 512,
        .detectorBlockSize = 512,
    };

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

        .beamformerIncoherentBeam = false,

        .detectorEnable = true,
        .detectorIntegrationSize = 1,
        .detectorNumberOfOutputPolarizations = 1,
        .detectorTransposedATPFrevOutput = true, // unnecessary AFTP == ATPF where T=P=1 

        // TODO: Review this calculation.
        .castBlockSize = 512,
        .channelizerBlockSize = 512,
        .phasorBlockSize = 512,
        .beamformerBlockSize = config.stepNumberOfTimeSamples,
        .detectorBlockSize = 512,
    };

    auto beamformRunner = Runner<Beamform>::New(config.numberOfWorkers, beamformConfig, false);

    // Instantiate search pipeline and runner.

    typename Search::Config searchConfig = {
        .inputDimensions = beamformRunner->getWorker().getOutputBuffer().dims(),
        .accumulateRate = 1,

        .searchMitigateDcSpike = true,
        .searchMinimumDriftRate = 0.0,
        .searchMaximumDriftRate = 10.0,
        .searchSnrThreshold = 10.0,

        .searchChannelBandwidthHz = reader.getChannelBandwidth(),
        .searchChannelTimespanS = 1.0 / reader.getChannelBandwidth(),
    };

    auto searchRunner = Runner<Search>::New(config.numberOfWorkers, searchConfig, false);

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

    std::unordered_map<U64, Vector<Device::CPU, F64>> stepJulianDateMap;
    std::unordered_map<U64, Vector<Device::CPU, F64>> stepDut1Map;
    std::unordered_map<U64, Vector<Device::CPU, U64>> stepFrequencyChannelOffsetMap;

    stepJulianDateMap.reserve(config.numberOfWorkers);
    stepDut1Map.reserve(config.numberOfWorkers);
    stepFrequencyChannelOffsetMap.reserve(config.numberOfWorkers);

    U64 stepCount = 0;
    U64 stepIncrement = channelizeConfig.accumulateRate*beamformConfig.accumulateRate*searchConfig.accumulateRate;
    U64 callbackStep = 0;
    U64 workerId = 0;

    while (Plan::Loop()) {
        readerRunner->enqueue([&](auto& worker){
            // Check if next runner has free slot.
            Plan::Available(channelizeRunner);

            // Read block of data.
            Plan::Compute(worker);

            // Transfer output data from this pipeline to the next runner.
            Plan::Accumulate(channelizeRunner, readerRunner,
                            worker.getStepOutputBuffer());
            stepCount += 1;

            stepJulianDateMap.insert({stepCount, worker.getStepOutputJulianDate()});
            stepDut1Map.insert({stepCount, worker.getStepOutputDut1()});
            stepFrequencyChannelOffsetMap.insert({stepCount, worker.getStepOutputFrequencyChannelOffset()});

            return stepCount;
        });

        channelizeRunner->enqueue([&](auto& worker) {
            // Check if next runner has free slot.
            Plan::Available(beamformRunner);

            // Try dequeue job from last runner. If unlucky, return.
            Plan::Dequeue(readerRunner, &callbackStep);

            // Increment progress bar.
            bar.set_progress(static_cast<float>(stepCount) /
                    reader.getNumberOfSteps() * 100);

            // Compute input data.
            Plan::Compute(worker);

            // Concatenate output data inside search pipeline.
            Plan::Accumulate(beamformRunner, channelizeRunner,
                             stepJulianDateMap[callbackStep],
                             stepDut1Map[callbackStep],
                             stepFrequencyChannelOffsetMap[callbackStep],
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
                             worker.getOutputBuffer());

            return callbackStep;
        });

        searchRunner->enqueue([&](auto& worker){
            // Try dequeue job from last runner. If unlucky, return.
            Plan::Dequeue(beamformRunner, &callbackStep);

            // If accumulation complete, write data to disk.
            Plan::Compute(worker);

            return callbackStep;
        });

        // Try to dequeue job from the final runner.
        if (searchRunner->dequeue(&callbackStep, &workerId)) {
            
            // TODO reimplement a link between the beamform-input/output/frequencyChannelOffset
            //  and the dedoppler output

            auto& searchWorker = searchRunner->getWorker(workerId);
            const std::vector<DedopplerHit>& hits = searchWorker.getOutputHits();

            if (hits.size() > 0) {
                BL_INFO("{} Hits:\n\tChannel Offset: {}", hits.size(), stepFrequencyChannelOffsetMap[callbackStep][0]);
                BL_INFO("\tJulian Date: {}", stepJulianDateMap[callbackStep][0]);

                for (DedopplerHit hit : hits) {
                    BL_INFO("\t{}", hit.toString());
                }
            }

            stepJulianDateMap.erase(callbackStep);
            stepDut1Map.erase(callbackStep);
            stepFrequencyChannelOffsetMap.erase(callbackStep);

            if ((callbackStep + stepIncrement) == reader.getNumberOfSteps()) {
                break;
            }
        }
    }

    // Gracefully destroy runners.

    readerRunner.reset();
    beamformRunner.reset();
    searchRunner.reset();

    return Result::SUCCESS;
}

}  // namespace Blade::CLI::Telescopes::ATA

#endif
