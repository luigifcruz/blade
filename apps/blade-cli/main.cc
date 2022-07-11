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

    // Read target coarse-channels.

    U64 coarse_channels = 32;

    app
        .add_option("-c,--coarse-channels", coarse_channels, "Coarse-channel ingest rate")
            ->default_val(32);

    //  Parse arguments.

    CLI11_PARSE(app, argc, argv);

    //  Print argument configurations.
    
    BL_INFO("Input GUPPI RAW File Path: {}", inputGuppiFile);
    BL_INFO("Input BFR5 File Path: {}", inputBfr5File);
    BL_INFO("Telescope: {}", telescope);
    BL_INFO("Mode: {}", mode);
    BL_INFO("Fine-time: {}", fine_time);
    BL_INFO("Coarse-channels: {}", coarse_channels);

    GuppiReader guppi = GuppiReader(
        {
            .filepath = inputGuppiFile,
            .blockSize = 32
        },
        {}
    );
    Bfr5Reader bfr5 = Bfr5Reader(inputBfr5File);
    if(guppi.getNumberOfAntenna() != bfr5.getDiminfo_nants()) {
        BL_FATAL("BFR5 and RAW files must specify the same number of antenna.");
        return 1;
    }
    if(guppi.getNumberOfPolarizations() != bfr5.getDiminfo_npol()) {
        BL_FATAL("BFR5 and RAW files must specify the same number of antenna.");
        return 1;
    }

    U64 channelizer_rate = (bfr5.getDiminfo_nants()*bfr5.getDiminfo_nchan())/guppi.getNumberOfFrequencyChannels();
    BL_INFO("Inferred channelizer rate: {}", channelizer_rate);
    if(channelizer_rate == 0) {
        BL_FATAL("Channelizer rate cannot be 0.");
        return 1;
    }
    
    if(coarse_channels != guppi.getNumberOfFrequencyChannels()) {
        BL_WARN(
            "Sub-band processing of the coarse-channels ({}/{}) is incompletely implemented: "
            "only the first sub-band is processed.",
            coarse_channels,
            guppi.getNumberOfFrequencyChannels()
        );
    }

    std::vector<std::complex<double>> antenna_weights(
        guppi.getNumberOfAntenna()*
        coarse_channels*channelizer_rate*
        guppi.getNumberOfPolarizations()
    );
    gather_antenna_weights_from_bfr5_cal_b(
        bfr5.getCalinfo_b(),
        guppi.getNumberOfAntenna(),
        bfr5.getDiminfo_nchan(),
        guppi.getNumberOfPolarizations(),
        0, // the first channel
        coarse_channels*channelizer_rate, // the number of channels
        antenna_weights.data() // [NANTS=slowest, number_of_channels, NPOL=fastest)]
    );

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
                        .numberOfPolarizations = guppi.getNumberOfPolarizations(),

                        .channelizerRate = channelizer_rate,

                        .beamformerBeams = bfr5.getDiminfo_nbeams(),
                        .enableIncoherentBeam = false,

                        .rfFrequencyHz = guppi.getBandwidthCenter(),
                        .channelBandwidthHz = guppi.getBandwidthOfChannel(),
                        .totalBandwidthHz = guppi.getBandwidthOfChannel()*coarse_channels,
                        .frequencyStartIndex = guppi.getChannelStartIndex(),
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
