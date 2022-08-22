#include <chrono>
#include <CLI/CLI.hpp>

#include "types.hh"
#include "telescopes/ata.hh"
#include "blade/pipelines/generic/file_reader.hh"

template<typename IT, typename OT>
inline const Result SetupTelescope(const CliConfig& config) {
    auto reader = Pipelines::Generic::FileReader<IT>({
        .inputGuppiFile = config.inputGuppiFile,
        .inputBfr5File = config.inputBfr5File,
        .stepNumberOfTimeSamples = config.stepNumberOfTimeSamples * 
                                   config.preBeamformerChannelizerRate,
        .stepNumberOfFrequencyChannels = config.stepNumberOfFrequencyChannels,
    });

    switch (config.telescope) {
        case TelescopeId::ATA:
            return SetupAta<IT, OT>(config, reader);
        default:
            BL_FATAL("Telescope not implemented yet.");
    }

    return Result::ERROR;
}

template<typename IT, typename OT>
inline const Result SetupProcessingPipeline(const CliConfig& config) {
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::milliseconds;

    auto t1 = high_resolution_clock::now();

    BL_CHECK((SetupTelescope<IT, OT>(config)));

    auto t2 = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(t2 - t1).count();
    BL_INFO("Processing finished in {} milliseconds!", duration);

    return Result::SUCCESS;
}

inline const Result CollectUserInput(int argc, char **argv, CliConfig& config) {
    CLI::App app("BLADE (Breakthrough Listen Accelerated DSP Engine) - Command Line Tool");

    // Read target telescope. 
    app
        .add_option("-t,--telescope", config.telescope, "Telescope ID (ATA, VLA, MEETKAT)")
            ->required()
            ->transform(CLI::CheckedTransformer(TelescopeMap, CLI::ignore_case));

    // Read target mode. 

    app
        .add_option("-m,--mode", config.mode, "Mode ID (MODE_B, MODE_A)")
            ->required()
            ->transform(CLI::CheckedTransformer(ModeMap, CLI::ignore_case));

    // Read input GUPPI RAW file.
    app
        .add_option("-i,--input,input", config.inputGuppiFile, "Input GUPPI RAW filepath")
            ->required()
            ->capture_default_str()
            ->run_callback_for_default();

    // Read input BFR5 file.
    app
        .add_option("-r,--recipe,recipe", config.inputBfr5File, "Input BFR5 filepath")
            ->required()
            ->capture_default_str()
            ->run_callback_for_default();

    // Read output GUPPI RAW filepath.
        app
        .add_option("-o,--output,output", config.outputGuppiFile, "Output GUPPI RAW filepath")
            ->required()
            ->capture_default_str()
            ->run_callback_for_default();

    // Read number of workers.
    app
        .add_option("-N,--number-of-workers", config.numberOfWorkers, "Number of workers")
            ->default_val(2);

    // Read target pre-beamformer channelizer rate.
    app
        .add_option("-c,--pre-beamformer-channelizer-rate", config.preBeamformerChannelizerRate, 
                "Pre-beamformer channelizer rate (FFT-size)")
            ->default_val(1024);

    // Read target step number of time samples.
    app
        .add_option("-T,--step-number-of-time-samples", config.stepNumberOfTimeSamples, 
                "Step number of time samples")
            ->default_val(32);

    // Read target step number of frequency channels.
    app
        .add_option("-C,--step-number-of-frequency-channels", config.stepNumberOfFrequencyChannels,
                "Step number of frequency channels")
            ->default_val(32);

    try {
        app.parse(argc, argv);
    } catch(const CLI::ParseError &e) {
        app.exit(e);
        return Result::ERROR;
    }

    // Print argument configurations.
    BL_INFO("Telescope: {}", config.telescope);
    BL_INFO("Mode: {}", config.mode);

    return Result::SUCCESS;
}

inline const Result StartCli(int argc, char **argv) {
    CliConfig config;

    BL_CHECK(CollectUserInput(argc, argv, config));
    BL_CHECK((SetupProcessingPipeline<CI8, CF32>(config)));

    return Result::SUCCESS;
}

int main(int argc, char **argv) {
    if (StartCli(argc, argv) != Result::SUCCESS) {
        return 1;
    }

    return 0;
}
