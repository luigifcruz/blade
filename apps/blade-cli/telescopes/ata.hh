#ifndef BLADE_CLI_TELESCOPES_ATA
#define BLADE_CLI_TELESCOPES_ATA

#include "types.hh"

#include "blade/runner.hh"
#include "blade/modules/guppi/writer.hh"

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

        .beamformerNumberOfAntennas = reader.getNumberOfAntennas(),
        .beamformerNumberOfFrequencyChannels = reader.getNumberOfFrequencyChannels(),
        .beamformerNumberOfTimeSamples = reader.getNumberOfTimeSamples(),
        .beamformerNumberOfPolarizations = reader.getNumberOfPolarizations(),
        .beamformerNumberOfBeams = reader.getNumberOfBeams(),
        .beamformerIncoherentBeam = false,

        .outputMemWidth = 8192,
        .outputMemPad = 0,

        .castBlockSize = 32,
        .channelizerBlockSize = cliConfig.stepNumberOfTimeSamples,
        .phasorBlockSize = 32,
        .beamformerBlockSize = cliConfig.stepNumberOfTimeSamples
    };

    auto runner = Runner<Pipeline>::New(cliConfig.numberOfWorkers, config);

    auto guppi_writer = Blade::Modules::Guppi::Writer<CF32>({
        .filepathStem = cliConfig.outputGuppiFile,
        .directio = 1,
        .numberOfAntennas = reader.getNumberOfAntennas(),
        .numberOfBeams = reader.getNumberOfBeams(),
        .numberOfFrequencyChannels = cliConfig.preBeamformerChannelizerRate*reader.getNumberOfFrequencyChannels(),
        .totalNumberOfFrequencyChannels = cliConfig.preBeamformerChannelizerRate*reader.getGuppi().getTotalNumberOfFrequencyChannels(),
        .numberOfTimeSamples = cliConfig.stepNumberOfTimeSamples, // post channelizer time (fine-time)
        .numberOfPolarizations = reader.getNumberOfPolarizations(),
    });

    guppi_writer.headerPut("OBSFREQ", reader.getObservationFrequency());
    guppi_writer.headerPut("OBSBW", reader.getChannelBandwidth()*guppi_writer.getTotalNumberOfFrequencyChannels()*guppi_writer.getNumberOfAspects());
    guppi_writer.headerPut("TBIN", cliConfig.preBeamformerChannelizerRate/reader.getChannelBandwidth());
    guppi_writer.headerPut("PKTIDX", 0);

    Vector<Device::CPU, CF32>* writer_batch_buffers[cliConfig.numberOfWorkers];
    for (U64 i = 0; i < cliConfig.numberOfWorkers; i++) {
        writer_batch_buffers[i] = new Vector<Device::CPU, CF32>(guppi_writer.getInputBatchSize());
        BL_INFO("Allocated Runner output buffer {}: {} ({} bytes)", 
                i, writer_batch_buffers[i]->size(), writer_batch_buffers[i]->size_bytes());
    }

    U64 buffer_idx = 0, job_idx = 0;
    U64 batch_idx, batch_offset;

    while (reader.run() == Result::SUCCESS) {
        const auto& res = runner->enqueue([&](auto& worker){
            worker.run(reader.getOutputEpochSeconds(), 0.0, 
                    reader.getOutput(), *writer_batch_buffers[buffer_idx]);
            return job_idx;
        });

        if (res) {
            job_idx++;
            buffer_idx = job_idx % cliConfig.numberOfWorkers;
        }

        if (runner->dequeue(&batch_idx)) {
            batch_offset = guppi_writer.getInputBatchOffset(
                batch_idx % guppi_writer.getNumberOfFrequencyChannelBatches()
            );
            memcpy(
                guppi_writer.getInput().data() + batch_offset,
                writer_batch_buffers[batch_idx % cliConfig.numberOfWorkers]->data(),
                writer_batch_buffers[batch_idx % cliConfig.numberOfWorkers]->size_bytes()
            );
            // write if that's the last batch-input for the batch
            if(
                (batch_idx + 1) % guppi_writer.getNumberOfFrequencyChannelBatches() == 0
            ) {
                guppi_writer.write();
                guppi_writer.headerPut("PKTIDX", guppi_writer.getNumberOfTimeSamples() * ((batch_idx+1) / guppi_writer.getNumberOfFrequencyChannelBatches()));
            }
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
