#ifndef BLADE_CLI_TELESCOPES_ATA
#define BLADE_CLI_TELESCOPES_ATA

#include "types.hh"

#include "blade/runner.hh"
#include "blade/pipelines/generic/file_writer.hh"

#ifdef BLADE_PIPELINE_ATA_MODE_A
#include "blade/pipelines/ata/mode_a.hh"
#endif

#ifdef BLADE_PIPELINE_ATA_MODE_B
#include "blade/pipelines/ata/mode_b.hh"
#endif

#ifdef BLADE_PIPELINE_ATA_MODE_H
#include "blade/pipelines/ata/mode_h.hh"
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

    // TODO: Replace this with direct copy to next worker's input buffer.
    Vector<Device::CPU, CF32>* writer_batch_buffers[cliConfig.numberOfWorkers];
    for (U64 i = 0; i < cliConfig.numberOfWorkers; i++) {
        writer_batch_buffers[i] = new Vector<Device::CPU, CF32>(writer.getStepInputBufferSize());
        BL_INFO("Allocated output buffer #{}: {} ({} bytes)", 
                i, writer_batch_buffers[i]->size(), writer_batch_buffers[i]->size_bytes());
    }

    U64 buffer_idx = 0, job_idx = 0;
    U64 batch_idx;

    while (reader.run() == Result::SUCCESS) {
        const auto& res = runner->enqueue([&](auto& worker){
            BL_CHECK_THROW(
                worker.run(reader.getStepOutputJulianDate(),
                           reader.getStepOutputDut1(), 
                           reader.getStepOutputBuffer(), 
                           *writer_batch_buffers[buffer_idx]));
            return job_idx;
        });

        if (res) {
            job_idx++;
            buffer_idx = job_idx % cliConfig.numberOfWorkers;
        }

        if (runner->dequeue(&batch_idx)) {
            // TODO: Implement this. LOL.
        }
    }

    runner.reset();

    return Result::SUCCESS;
}

template<typename IT, typename OT>
inline const Result SetupAta(const CliConfig& config,
                             Pipelines::Generic::FileReader<IT>& reader) {
    switch (config.mode) {
#ifdef BLADE_PIPELINE_ATA_MODE_A
#endif
#ifdef BLADE_PIPELINE_ATA_MODE_B
        case ModeId::MODE_B:
            return SetupAtaModeB<IT, OT>(config, reader);
#endif
#ifdef BLADE_PIPELINE_ATA_MODE_H
#endif
        default:
            BL_FATAL("This ATA mode is not implemented yet.");
    }

    return Result::ERROR;
}

#endif
