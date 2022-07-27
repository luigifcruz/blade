#include <CLI/CLI.hpp>
#include <iostream>
#include <string>

#include "blade/base.hh"
#include "blade/logger.hh"
#include "blade/runner.hh"
#include "blade/pipelines/ata/mode_b.hh"
#include "blade/pipelines/generic/file_reader.hh"

using namespace Blade;

typedef enum {
    ATA,
    VLA,
    MEERKAT,
} TelescopeId;

typedef enum {
    MODE_B,
    MODE_A,
} ModeId;

typedef struct {
    ModeId mode;
    TelescopeId telescope;
    std::string inputGuppiFile;
    std::string inputBfr5File;
    U64 preBeamformerChannelizerRate;
    U64 stepNumberOfTimeSamples;
    U64 stepNumberOfFrequencyChannels;
} CliConfig;

using CLIPipeline = Blade::Pipelines::ATA::ModeB<CF32>;
static std::unique_ptr<Runner<CLIPipeline>> runner;

int main(int argc, char **argv) {

    CliConfig cliConfig;

    CLI::App app("BLADE (Breakthrough Listen Accelerated DSP Engine) Command Line Tool");

    // Read target telescope. 

    std::map<std::string, TelescopeId> telescopeMap = {
        {"ATA",     TelescopeId::ATA}, 
        {"VLA",     TelescopeId::VLA},
        {"MEERKAT", TelescopeId::MEERKAT}
    };

    app
        .add_option("-t,--telescope", cliConfig.telescope, "Telescope ID (ATA, VLA, MEETKAT)")
            ->required()
            ->transform(CLI::CheckedTransformer(telescopeMap, CLI::ignore_case));

    // Read target mode. 

    std::map<std::string, ModeId> modeMap = {
        {"MODE_B",     ModeId::MODE_B}, 
        {"MODE_A",     ModeId::MODE_A},
        {"B",          ModeId::MODE_B}, 
        {"A",          ModeId::MODE_A},
    };

    app
        .add_option("-m,--mode", cliConfig.mode, "Mode ID (MODE_B, MODE_A)")
            ->required()
            ->transform(CLI::CheckedTransformer(modeMap, CLI::ignore_case));

    // Read input GUPPI RAW file.

    app
        .add_option("-i,--input,input", cliConfig.inputGuppiFile, "Input GUPPI RAW filepath")
            ->required()
            ->capture_default_str()
            ->run_callback_for_default();

    // Read input BFR5 file.

    app
        .add_option("-r,--recipe,recipe", cliConfig.inputBfr5File, "Input BFR5 filepath")
            ->required()
            ->capture_default_str()
            ->run_callback_for_default();

    // Read target pre-beamformer channelizer rate.

    app
        .add_option("-c,--pre-beamformer-channelizer-rate", cliConfig.preBeamformerChannelizerRate, 
                "Pre-beamformer channelizer rate (FFT-size)")
            ->default_val(1024);

    // Read target step number of time samples.

    app
        .add_option("-T,--step-number-of-time-samples", cliConfig.stepNumberOfTimeSamples, 
                "Step number of time samples")
            ->default_val(32);

    // Read target step number of frequency channels.

    app
        .add_option("-C,--step-number-of-frequency-channels", cliConfig.stepNumberOfFrequencyChannels,
                "Step number of frequency channels")
            ->default_val(32);

    // Parse arguments.

    CLI11_PARSE(app, argc, argv);

    // Print argument configurations.
    
    BL_INFO("Input GUPPI RAW File Path: {}", cliConfig.inputGuppiFile);
    BL_INFO("Input BFR5 File Path: {}", cliConfig.inputBfr5File);
    BL_INFO("Telescope: {}", cliConfig.telescope);
    BL_INFO("Mode: {}", cliConfig.mode);
    BL_INFO("Pre-beamformer channelizer rate: {}", cliConfig.preBeamformerChannelizerRate);

    auto reader = Pipelines::Generic::FileReader<CI8>({
        .inputGuppiFile = cliConfig.inputGuppiFile,
        .inputBfr5File = cliConfig.inputBfr5File,
        .stepNumberOfTimeSamples = cliConfig.stepNumberOfTimeSamples * 
                                   cliConfig.preBeamformerChannelizerRate,
        .stepNumberOfFrequencyChannels = cliConfig.stepNumberOfFrequencyChannels,
    });

    const int numberOfWorkers = 1;
    switch (cliConfig.telescope) {
        case TelescopeId::ATA:
            switch (cliConfig.mode) {
                case ModeId::MODE_A:
                    BL_ERROR("Unsupported mode for ATA selected. WIP.");
                    break;
                case ModeId::MODE_B:
                    CLIPipeline::Config config = {
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
                    runner = Runner<CLIPipeline>::New(numberOfWorkers, config);
                    break;
            }
            break;
        default:
            BL_ERROR("Unsupported telescope selected. WIP");
            return 1;
    }

    Vector<Device::CPU, CF32>* output_buffers[numberOfWorkers];
    for (int i = 0; i < numberOfWorkers; i++) {
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
            buffer_idx = (buffer_idx + 1) % numberOfWorkers;
        }

        if (runner->dequeue(nullptr)) {
            job_idx++;
        }
    }

    BL_INFO("Processing finished!");

    runner.reset();

    return 0;
}
