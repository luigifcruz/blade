#ifndef BLADE_CLI_TELESCOPES_ATA
#define BLADE_CLI_TELESCOPES_ATA

#include "types.hh"

#include "blade/runner.hh"
#include "blade/pipelines/generic/file_writer.hh"
#include "blade/utils/progressbar.hh"

#ifdef BLADE_PIPELINE_ATA_MODE_A
#include "blade/pipelines/ata/mode_a.hh"
#endif

#ifdef BLADE_PIPELINE_ATA_MODE_B
#include "blade/pipelines/ata/mode_b.hh"
#endif

template<typename IT, typename OT>
inline const Result SetupAtaModeB(const CliConfig& cliConfig, 
                                  Pipelines::Generic::FileReader<IT>& reader) {
    using Pipeline = Pipelines::ATA::ModeB<OT>;

    typename Pipeline::Config config = {
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

        .outputMemWidth = 8192,
        .outputMemPad = 0,

        .castBlockSize = 32,
        .channelizerBlockSize = cliConfig.stepNumberOfTimeSamples,
        .phasorBlockSize = 32,
        .beamformerBlockSize = cliConfig.stepNumberOfTimeSamples
    };

    auto runner = Runner<Pipeline>::New(cliConfig.numberOfWorkers, config);

    auto writer = Pipelines::Generic::FileWriter<OT>({
        .outputGuppiFile = cliConfig.outputGuppiFile,
        .directio = true,

        .stepNumberOfBeams = reader.getStepNumberOfBeams(),
        .stepNumberOfAntennas = reader.getStepNumberOfAntennas(),
        .stepNumberOfFrequencyChannels = cliConfig.preBeamformerChannelizerRate * reader.getStepNumberOfFrequencyChannels(),
        .stepNumberOfTimeSamples = cliConfig.stepNumberOfTimeSamples,
        .stepNumberOfPolarizations = reader.getStepNumberOfPolarizations(),

        .totalNumberOfFrequencyChannels = cliConfig.preBeamformerChannelizerRate * reader.getTotalNumberOfFrequencyChannels(),
    });

    writer.headerPut("OBSFREQ", reader.getObservationFrequency());
    writer.headerPut("OBSBW", reader.getChannelBandwidth() * 
                                    writer.getTotalNumberOfFrequencyChannels() * 
                                    writer.getStepNumberOfAntennas() * 
                                    writer.getStepNumberOfBeams());
    writer.headerPut("TBIN", cliConfig.preBeamformerChannelizerRate / reader.getChannelBandwidth());
    writer.headerPut("PKTIDX", 0);

    BL_INFO("{} {} {}", reader.getStepOutputBufferSize(), reader.getTotalOutputBufferSize(), reader.getTotalOutputBufferSize() / reader.getStepOutputBufferSize());

    U64 counter = 0;
    while (true) {
        if (reader.run() == Result::SUCCESS) {
            const auto& res = runner->enqueue([&](auto& worker){
                BL_CHECK_THROW(
                    worker.run(reader.getStepOutputJulianDate(),
                               reader.getStepOutputDut1(), 
                               reader.getStepOutputBuffer(), 
                               writer));
                return 0;
            });

            if (!res) {
                BL_FATAL("Runner I/O error.");
                BL_CHECK_THROW(Result::ERROR);
            }
        } else {
            if (runner->empty()) {
                break;
            }
        }

        // TODO: This is not working. I need to know the reader number of steps.

        if (runner->dequeue(nullptr)) {
            counter++;
            if (writer.accumulationComplete()) {
                BL_CHECK_THROW(writer.run());
            }
        }
    }

    BL_INFO("{} {} {}", counter, writer.getNumberOfSteps(), reader.getNumberOfSteps())

    runner.reset();

    return Result::SUCCESS;
}

template<typename IT, typename OT>
inline const Result SetupAta(const CliConfig& config,
                             Pipelines::Generic::FileReader<IT>& reader) {
    switch (config.mode) {
#if defined(BLADE_PIPELINE_ATA_MODE_B)
        case ModeId::MODE_B:
            return SetupAtaModeB<IT, OT>(config, reader);
#endif
#if defined(BLADE_PIPELINE_ATA_MODE_B) && defined(BLADE_PIPELINE_GENERIC_MODE_H)
#endif
        default:
            BL_FATAL("This ATA mode is not implemented yet.");
    }

    return Result::ERROR;
}

#endif
