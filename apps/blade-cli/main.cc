#include <CLI/CLI.hpp>
#include <iostream>
#include <string>

#include "blade/base.hh"
#include "blade/logger.hh"
#include "blade/runner.hh"
#include "blade/modules/guppi/reader.hh"
#include "blade/modules/bfr5/reader.hh"
#include "blade/pipelines/ata/mode_b.hh"

typedef enum {
    ATA,
    VLA,
    MEERKAT,
} TelescopeID;

typedef enum {
    MODE_B,
    MODE_A,
} ModeID;

using namespace Blade;

using GuppiReader = Blade::Modules::Guppi::Reader<CI8>;
using Bfr5Reader = Blade::Modules::Bfr5::Reader;
using CLIPipeline = Blade::Pipelines::ATA::ModeB<CF32>;
static std::unique_ptr<Runner<CLIPipeline>> runner;

int main(int argc, char **argv) {

    CLI::App app("BLADE (Breakthrough Listen Accelerated DSP Engine) Command Line Tool");

    //  Read target telescope. 

    TelescopeID telescope = TelescopeID::ATA;

    std::map<std::string, TelescopeID> telescopeMap = {
        {"ATA",     TelescopeID::ATA}, 
        {"VLA",     TelescopeID::VLA},
        {"MEERKAT", TelescopeID::MEERKAT}
    };

    app
        .add_option("-t,--telescope", telescope, "Telescope ID (ATA, VLA, MEETKAT)")
            ->required()
            ->transform(CLI::CheckedTransformer(telescopeMap, CLI::ignore_case));

    //  Read target mode. 

    ModeID mode = ModeID::MODE_B;

    std::map<std::string, ModeID> modeMap = {
        {"MODE_B",     ModeID::MODE_B}, 
        {"MODE_A",     ModeID::MODE_A},
        {"B",          ModeID::MODE_B}, 
        {"A",          ModeID::MODE_A},
    };

    app
        .add_option("-m,--mode", mode, "Mode ID (MODE_B, MODE_A)")
            ->required()
            ->transform(CLI::CheckedTransformer(modeMap, CLI::ignore_case));

    //  Read input GUPPI RAW file.

    std::string inputGuppiFile;

    app
        .add_option("-i,--input,input", inputGuppiFile, "Input GUPPI RAW filepath")
            ->required()
            ->capture_default_str()
            ->run_callback_for_default();

    //  Read input BFR5 file.

    std::string inputBfr5File;

    app
        .add_option("-r,--recipe,recipe", inputBfr5File, "Input BFR5 filepath")
            ->required()
            ->capture_default_str()
            ->run_callback_for_default();

    // Read target fine-time.

    U64 fine_time = 32;

    app
        .add_option("-T,--fine-time", fine_time, "Number of fine-timesamples")
            ->default_val(32);

    // Read target channelizer-rate.

    U64 channelizer_rate = 1024;

    app
        .add_option("-c,--channelizer", channelizer_rate, "Channelizer (FFT) rate")
            ->default_val(1024);

    // Read target coarse-channels.

    U64 coarse_channels = 32;

    app
        .add_option("-C,--coarse-channels", coarse_channels, "Coarse-channel ingest rate")
            ->default_val(32);

    //  Parse arguments.

    CLI11_PARSE(app, argc, argv);

    //  Print argument configurations.
    
    BL_INFO("Input GUPPI RAW File Path: {}", inputGuppiFile);
    BL_INFO("Input BFR5 File Path: {}", inputBfr5File);
    BL_INFO("Telescope: {}", telescope);
    BL_INFO("Mode: {}", mode);
    BL_INFO("Fine-time: {}", fine_time);
    BL_INFO("Channelizer-rate: {}", channelizer_rate);
    BL_INFO("Coarse-channels: {}", coarse_channels);

    GuppiReader guppi = GuppiReader(
        {
            .filepath = inputGuppiFile,
            .blockSize = 32
        },
        {}
    );
    Bfr5Reader bfr5 = Bfr5Reader(inputBfr5File);

    // Argument-conditional Pipeline
    const int numberOfWorkers = 1;
    switch (telescope) {
        case TelescopeID::ATA:
            switch (mode) {
                case ModeID::MODE_A:
                    BL_ERROR("Unsupported mode for ATA selected. WIP.");
                    break;
                case ModeID::MODE_B:
                    CLIPipeline::Config config = {
                        .numberOfAntennas = guppi.getNumberOfAntenna(),
                        .numberOfFrequencyChannels = coarse_channels,
                        .numberOfTimeSamples = fine_time*channelizer_rate,
                        .numberOfPolarizations = guppi.getNumberOfOutputPolarizations(),

                        .channelizerRate = channelizer_rate,

                        .beamformerBeams = bfr5.getDiminfo_nbeams(),
                        .enableIncoherentBeam = false,

                        .rfFrequencyHz = 6500.125*1e6,
                        .channelBandwidthHz = 0.5e6,
                        .totalBandwidthHz = 1.024e9,
                        .frequencyStartIndex = 352,
                        .referenceAntennaIndex = 0,
                        .arrayReferencePosition = bfr5.getTelinfo_lla(),
                        .boresightCoordinate = bfr5.getObsinfo_phase_center(),
                        .antennaPositions = bfr5.getTelinfo_antenna_positions(),
                        .antennaCalibrations = {},
                        .beamCoordinates = bfr5.getBeaminfo_coordinates(),

                        .outputMemWidth = 8192,
                        .outputMemPad = 0,

                        .castBlockSize = 32,
                        .channelizerBlockSize = fine_time,
                        .phasorsBlockSize = 32,
                        .beamformerBlockSize = fine_time
                    };
                    config.antennaCalibrations.resize(
                        config.numberOfAntennas *
                        config.numberOfFrequencyChannels *
                        config.channelizerRate *
                        config.numberOfPolarizations
                    );
                    runner = Runner<CLIPipeline>::New(numberOfWorkers, config);
                    break;
            }
            break;
        default:
            BL_ERROR("Unsupported telescope selected. WIP");
            return 1;
    }

    return 0;
}
