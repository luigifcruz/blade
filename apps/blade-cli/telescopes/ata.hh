#ifndef BLADE_CLI_TELESCOPES_ATA
#define BLADE_CLI_TELESCOPES_ATA

#include "types.hh"

#include "blade/runner.hh"

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

    Vector<Device::CPU, CF32>* output_buffers[cliConfig.numberOfWorkers];
    for (U64 i = 0; i < cliConfig.numberOfWorkers; i++) {
        output_buffers[i] = new Vector<Device::CPU, CF32>(runner->getWorker().getOutputSize());
        BL_INFO("Allocated Runner output buffer {}: {} ({} bytes)", 
                i, output_buffers[i]->size(), output_buffers[i]->size_bytes());
    }

    U64 buffer_idx = 0;
    U64 job_idx = 0;

    while (reader.run() == Result::SUCCESS) {
        const auto& res = runner->enqueue([&](auto& worker){
            worker.run(reader.getOutputJulianDate(), 0.0, 
                    reader.getOutput(), *output_buffers[buffer_idx]);
            return job_idx;
        });

        if (res) {
            buffer_idx = (buffer_idx + 1) % cliConfig.numberOfWorkers;
        }

        if (runner->dequeue(nullptr)) {
            job_idx++;
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
