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

using CLIPipeline = Blade::Pipelines::ATA::ModeB<CF32>;
static std::unique_ptr<Runner<CLIPipeline>> runner;

int main(int argc, char **argv) {

    CLI::App app("BLADE (Breakthrough Listen Accelerated DSP Engine) Command Line Tool");

    // Read target telescope. 

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

    // Read target mode. 

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

    // Read input GUPPI RAW file.

    std::string inputGuppiFile;

    app
        .add_option("-i,--input,input", inputGuppiFile, "Input GUPPI RAW filepath")
            ->required()
            ->capture_default_str()
            ->run_callback_for_default();

    // Read input BFR5 file.

    std::string inputBfr5File;

    app
        .add_option("-r,--recipe,recipe", inputBfr5File, "Input BFR5 filepath")
            ->required()
            ->capture_default_str()
            ->run_callback_for_default();

    // Read target step number of time samples.

    U64 numberOfTimeSamples = 32;

    app
        .add_option("-T,--time-samples", numberOfTimeSamples, "Step number of time samples")
            ->default_val(32);

    // Read target pre-channelizer rate.

    U64 preChannelizerRate = 1024;

    app
        .add_option("-c,--pre-channelizer-rate", preChannelizerRate, "Pre-channelizer rate (FFT-size)")
            ->default_val(1024);

    // Read target step number of frequency channels.

    U64 numberOfFrequencyChannels = 32;

    app
        .add_option("-C,--frequency-channels", numberOfFrequencyChannels, "Step number of frequency channels")
            ->default_val(32);

    // TODO: How the frequency channel offset is handled?

    // Parse arguments.

    CLI11_PARSE(app, argc, argv);

    // Print argument configurations.
    
    BL_INFO("Input GUPPI RAW File Path: {}", inputGuppiFile);
    BL_INFO("Input BFR5 File Path: {}", inputBfr5File);
    BL_INFO("Telescope: {}", telescope);
    BL_INFO("Mode: {}", mode);
    BL_INFO("Pre-channelizer rate: {}", preChannelizerRate);

    auto guppi = Blade::Modules::Guppi::Reader<CI8>({
        .filepath = inputGuppiFile,
        .stepNumberOfTimeSamples = numberOfTimeSamples * preChannelizerRate,  // TODO: Larger than the original file. Error? 
        .stepNumberOfFrequencyChannels = numberOfFrequencyChannels,
    }, {});

    auto bfr5 = Blade::Modules::Bfr5::Reader({
        .filepath = inputBfr5File,
    }, {});

    if (guppi.getTotalNumberOfAntennas() != bfr5.getTotalNumberOfAntennas()) {
        BL_FATAL("BFR5 and RAW files must specify the same number of antenna.");
        return 1;
    }

    if (guppi.getTotalNumberOfFrequencyChannels() != bfr5.getTotalNumberOfFrequencyChannels()) {
        BL_FATAL("BFR5 and RAW files must specify the same number of frequency channels.");
        return 1;
    }

    if (guppi.getTotalNumberOfPolarizations() != bfr5.getTotalNumberOfPolarizations()) {
        BL_FATAL("BFR5 and RAW files must specify the same number of antenna.");
        return 1;
    }

    // TODO: How about Time Samples?
    
    if (numberOfFrequencyChannels != guppi.getTotalNumberOfFrequencyChannels()) {
        BL_WARN(
            "Sub-band processing of the coarse-channels ({}/{}) is incompletely implemented: "
            "only the first sub-band is processed.",
            numberOfFrequencyChannels,
            guppi.getTotalNumberOfFrequencyChannels());
    }

    const int numberOfWorkers = 1;
    switch (telescope) {
        case TelescopeID::ATA:
            switch (mode) {
                case ModeID::MODE_A:
                    BL_ERROR("Unsupported mode for ATA selected. WIP.");
                    break;
                case ModeID::MODE_B:
                    CLIPipeline::Config config = {
                        .numberOfAntennas = guppi.getNumberOfAntennas(),
                        .numberOfFrequencyChannels = numberOfFrequencyChannels,
                        .numberOfTimeSamples = guppi.getNumberOfTimeSamples(),
                        .numberOfPolarizations = guppi.getNumberOfPolarizations(),

                        .preChannelizerRate = preChannelizerRate,

                        .beamformerBeams = bfr5.getTotalNumberOfBeams(),
                        .enableIncoherentBeam = false,

                        .rfFrequencyHz = guppi.getBandwidthCenter(),
                        .channelBandwidthHz = guppi.getBandwidthOfChannel(),
                        .totalBandwidthHz = guppi.getBandwidthOfChannel() * numberOfFrequencyChannels,
                        .frequencyStartIndex = guppi.getChannelStartIndex(),
                        .referenceAntennaIndex = 0,
                        .arrayReferencePosition = bfr5.getReferencePosition(),
                        .boresightCoordinate = bfr5.getBoresightCoordinate(),
                        .antennaPositions = bfr5.getAntennaPositions(),
                        .antennaCalibrations = bfr5.getAntennaCalibrations(numberOfFrequencyChannels, 
                                preChannelizerRate),
                        .beamCoordinates = bfr5.getBeamCoordinates(),

                        .outputMemWidth = 8192,
                        .outputMemPad = 0,

                        .castBlockSize = 32,
                        .channelizerBlockSize = numberOfTimeSamples,
                        .phasorsBlockSize = 32,
                        .beamformerBlockSize = numberOfTimeSamples
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
        BL_INFO(
            "Allocated Runner output buffer {}: {} ({} bytes)",
            i,
            output_buffers[i]->size(),
            output_buffers[i]->size_bytes()
        );
    }

    U64 buffer_idx = 0;
    U64 job_idx = 0;

    while(guppi.canRead()) {
        if (runner->enqueue(
        [&](auto& worker){
            worker.run(
                guppi.getBlockEpochSeconds(guppi.getNumberOfTimeSamples()/2),
                0.0,
                guppi.getOutput(),
                *output_buffers[buffer_idx]
            );
            return job_idx;
        }
        )) {
            buffer_idx = (buffer_idx + 1) % numberOfWorkers;
        }

        if (runner->dequeue(nullptr)) {
            job_idx++;
        }
    }

    runner.reset();

    return 0;
}
