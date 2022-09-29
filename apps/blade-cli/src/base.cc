#define BL_LOG_DOMAIN "CLI"

#include "blade-cli/base.hh"

namespace Blade::CLI {

const Result SetupTelescope(const Config& config) {
    // Setup the telescope system.

    switch (config.telescope) {
        case TelescopeId::ATA:
            return Telescopes::ATA::Setup(config);
        case TelescopeId::GENERIC:
            return Telescopes::Generic::Setup(config);
        case TelescopeId::VLA:
            return Telescopes::VLA::Setup(config);
        default:
            BL_FATAL("Telescope not implemented yet.");
    }

    return Result::ERROR;
}

const Result SetupProcessingPipeline(const Config& config) {
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::milliseconds;

    auto t1 = high_resolution_clock::now();

    BL_CHECK((SetupTelescope(config)));

    auto t2 = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(t2 - t1).count();
    BL_INFO("Setup and processing finished in {} milliseconds!", duration);

    return Result::SUCCESS;
}

const Result CollectUserInput(int argc, char **argv, Config& config) {
    ::CLI::App app("BLADE (Breakthrough Listen Accelerated DSP Engine) - Command Line Tool");

    // Read target telescope. 
    app
        .add_option("-t,--telescope", config.telescope, "Telescope ID (ATA, VLA, MEETKAT, GENERIC)")
            ->required()
            ->transform(::CLI::CheckedTransformer(TelescopeMap, ::CLI::ignore_case));

    // Read target mode. 

    app
        .add_option("-m,--mode", config.mode, "Mode ID (MODE_B, MODE_H, MODE_BH)")
            ->required()
            ->transform(::CLI::CheckedTransformer(ModeMap, ::CLI::ignore_case));

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
        .add_option("-o,--output,output", config.outputFile, "Output filepath")
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

    // Read input type format.
    app
        .add_option("--input-type", config.inputType, "Input type format (CI8, CF16, CF32, I8, F16, F32)")
            ->required()
            ->transform(::CLI::CheckedTransformer(TypeMap, ::CLI::ignore_case));

    // Read output type format.
    app
        .add_option("--output-type", config.outputType, "Output type format (CI8, CF16, CF32, I8, F16, F32)")
            ->required()
            ->transform(::CLI::CheckedTransformer(TypeMap, ::CLI::ignore_case));

    try {
        app.parse(argc, argv);
    } catch(const ::CLI::ParseError &e) {
        app.exit(e);
        return Result::ERROR;
    }

    // Print our beloved E.T.
    BL_LOG_PRINT_ET();

    return Result::SUCCESS;
}

const Result Start(int argc, char **argv) {
    Config config;

    BL_CHECK(CollectUserInput(argc, argv, config));
    BL_CHECK((SetupProcessingPipeline(config)));

    return Result::SUCCESS;
}

}  // namespace Blade::CLI
