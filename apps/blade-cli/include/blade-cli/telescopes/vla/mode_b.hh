#ifndef BLADE_CLI_TELESCOPES_VLA_MODE_HB_HH
#define BLADE_CLI_TELESCOPES_VLA_MODE_HB_HH

#include "blade-cli/types.hh"

#include "blade/plan.hh"
#include "blade/runner.hh"
#include "blade/utils/indicators.hh"
#include "blade/pipelines/vla/mode_b.hh"
#include "blade/modules/phasor/ata.hh"
#include "blade/pipelines/generic/file_reader.hh"
#include "blade/pipelines/generic/file_writer.hh"

using namespace indicators;

namespace Blade::CLI::Telescopes::VLA {

template<typename IT, typename OT>
inline const Result ModeB(const Config& config) {
    // Define some types.
    using Reader = Pipelines::Generic::FileReader<IT>;
    using Compute = Pipelines::VLA::ModeB<IT, OT>;
    using Writer = Pipelines::Generic::FileWriter<Modules::Filterbank::Writer<OT>, OT>;

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

    auto phasorCoeffDims = reader.getStepOutputDims() * ArrayTensorDimensions({
        .A = 1,
        .F = config.preBeamformerChannelizerRate,
        .T = 0,
        .P = 1,
    });
    auto phasorAntennaCoeffs = ArrayTensor<Device::CPU, CF64>(reader.getAntennaCoefficientsDims(1));
    reader.fillAntennaCoefficients(1, phasorAntennaCoeffs);

    // Instantiate compute pipeline and runner.

    // TODO: less static hardware limit `1024`
    const auto timeRelatedBlockSize = config.stepNumberOfTimeSamples > 1024 ? 1024 : config.stepNumberOfTimeSamples;

    typename Compute::Config computeConfig = {
        .inputDimensions = reader.getStepOutputDims(),

        .preBeamformerChannelizerRate = config.preBeamformerChannelizerRate,

        .phasorChannelZeroFrequencyHz = reader.getObservationFrequency() - (reader.getTotalBandwidth() / 2.0),
        .phasorChannelBandwidthHz = reader.getChannelBandwidth(),
        .phasorFrequencyStartIndex = reader.getChannelStartIndex(),

        .phasorAntennaCoefficients = ArrayTensor<Device::CPU, CF64>(phasorAntennaCoeffs),
        .phasorBeamAntennaDelays = PhasorTensor<Device::CPU, F64>(reader.getBeamAntennaDelays(), reader.getBeamAntennaDelayDims()),
        .phasorDelayTimes = Vector<Device::CPU, F64>(reader.getDelayTimes(), {reader.getNumberOfDelayTimes()}),

        .beamformerNumberOfBeams = reader.getBeamAntennaDelayDims().numberOfBeams(),
        .beamformerIncoherentBeam = true,

        .detectorEnable = true,
        .detectorIntegrationSize = 1,
        .detectorNumberOfOutputPolarizations = 1,

        // TODO: Review this calculation.
        .castBlockSize = 32,
        .channelizerBlockSize = timeRelatedBlockSize,
        .phasorBlockSize = 32,
        .beamformerBlockSize = timeRelatedBlockSize,
        .detectorBlockSize = 32,
    };

    auto computeRunner = Runner<Compute>::New(config.numberOfWorkers, computeConfig, false);

    // Instantiate writer pipeline and runner.

    typename Writer::Config writerConfig = {
        .writerConfig = {
            .filepath = config.outputFile,
            
            .machineId = 0,
            .telescopeId = 0,
            .baryCentric = 1,
            .pulsarCentric = 1,
            .sourceCoordinate = reader.getBoresightCoordinate(),
            .azimuthStart = 0.0,
            .zenithStart = 0.0,
            .firstChannelCenterFrequency = (-1*reader.getObservationFrequency())
                - (-1*reader.getChannelBandwidth())*(readerTotalOutputDims.numberOfFrequencyChannels()-1)
                    /2,
            .channelBandwidthHz = -1*reader.getChannelBandwidth(),
            .julianDateStart = reader.getJulianDateOfLastReadBlock(),
            .numberOfIfChannels = (I32) computeRunner->getWorker().getOutputBuffer().dims().numberOfPolarizations(),
            .source_name = "Unknown",
            .rawdatafile = config.inputGuppiFile,

            .numberOfInputFrequencyChannelBatches = readerTotalOutputDims.numberOfFrequencyChannels() / computeConfig.inputDimensions.numberOfFrequencyChannels(),
        },
        .inputDimensions = computeRunner->getWorker().getOutputBuffer().dims(),
        .transposeBTPF = true,
        .accumulateRate = readerTotalOutputDims.numberOfFrequencyChannels() / computeConfig.inputDimensions.numberOfFrequencyChannels(),
    };

    auto writerRunner = Runner<Writer>::New(1, writerConfig, false);

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

}  // namespace Blade::CLI::Telescopes::VLA

#endif
