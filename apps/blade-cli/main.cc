#include <CLI/CLI.hpp>
#include <iostream>
#include <string>

#include "blade/base.hh"
#include "blade/logger.hh"
#include "blade/runner.hh"
#include "blade/pipelines/ata/mode_b.hh"
#include "blade/pipelines/ata/mode_h.hh"

extern "C" {
    #include "guppiraw.h"
}

typedef enum {
    ATA,
    VLA,
    MEERKAT,
} TelescopeID;

typedef enum {
    MODE_B,
    MODE_A,
} ModeID;

typedef struct {
  int nants;
} guppiraw_block_meta_t;

void guppiraw_parse_block_meta(char* entry, void* block_meta_void) {
  guppiraw_block_meta_t* block_meta = (guppiraw_block_meta_t*) block_meta_void;
  switch (((uint64_t*)entry)[0]) {
    case KEY_UINT64_ID_LE('N','A','N','T','S',' ',' ',' '):
      hgeti4(entry, "NANTS", &block_meta->nants);
      break;
    default:
      break;
  }
}

using namespace Blade;

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

    //  Read input file.

    std::string inputFile;

    app
        .add_option("-i,--input,input", inputFile, "Input filepath")
            ->required()
            ->capture_default_str()
            ->run_callback_for_default();

    // Read target beams.

    U64 beams = 8;

    app
        .add_option("-b,--beams", beams, "Number of beams")
            ->default_val(8);

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
    
    BL_INFO("Input File Path: {}", inputFile);
    BL_INFO("Telescope: {}", telescope);
    BL_INFO("Mode: {}", mode);
    BL_INFO("Beams: {}", beams);
    BL_INFO("Fine-time: {}", fine_time);
    BL_INFO("Channelizer-rate: {}", channelizer_rate);
    BL_INFO("Coarse-channels: {}", coarse_channels);

    guppiraw_iterate_info_t gr_iterate = {0};
    gr_iterate.file_info.block_info.header_user_data = malloc(sizeof(guppiraw_block_meta_t));
    gr_iterate.file_info.block_info.header_entry_callback = guppiraw_parse_block_meta;

    if (guppiraw_iterate_open_stem(inputFile.c_str(), &gr_iterate)) {
        BL_ERROR("Could not open: {}.{:04d}.raw\n", gr_iterate.stempath, gr_iterate.fileenum);
        return 1;
    }
    guppiraw_datashape_t *datashape = &gr_iterate.file_info.block_info.datashape;
    BL_INFO("GUPPI RAW file datashape: [{}, {}, {}, {}, CI{}] ({} bytes)",
        ((guppiraw_block_meta_t*)gr_iterate.file_info.block_info.header_user_data)->nants,
        datashape->n_obschan/((guppiraw_block_meta_t*)gr_iterate.file_info.block_info.header_user_data)->nants,
        datashape->n_time,
        datashape->n_pol,
        datashape->n_bit,
        datashape->block_size
    );

    // Argument-conditional Pipeline
    // Runner<Blade::Pipeline>* runner;
    // const int numberOfWorkers = 1;
    // switch (telescope) {
    //     case TelescopeID::ATA:
    //         switch (mode) {
    //             case ModeID::MODE_B:
    //                 using CLIPipeline = Blade::Pipelines::ATA::ModeB<CF32>;
    //                 CLIPipeline::Config config = {
    //                     .numberOfAntennas = 1,
    //                     .numberOfFrequencyChannels = coarse_channels,
    //                     .numberOfTimeSamples = fine_time*channelizer_rate,
    //                     .numberOfPolarizations = datashape->n_pol,

    //                     .channelizerRate = channelizer_rate,

    //                     .beamformerBeams = beams,

    //                     .outputMemWidth = 8192,
    //                     .outputMemPad = 0,

    //                     .castBlockSize = 32,
    //                     .channelizerBlockSize = fine_time,
    //                     .phasorsBlockSize = 32,
    //                     .beamformerBlockSize = fine_time
    //                 };
    //                 runner = new Runner<CLIPipeline>(numberOfWorkers, config);
    //                 break;
    //         }
    //         break;
    //     default:
    //         BL_ERROR("Unsupported telescope selected.");
    //         return 1;
    // }

    return 0;
}
