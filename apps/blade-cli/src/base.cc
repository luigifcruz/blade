#define BL_LOG_DOMAIN "CLI"

#include "blade-cli/base.hh"

namespace Blade::CLI {

const Result SetupTelescope(int argc, char **argv, const Config& config) {
    // Setup the telescope system.
    Telescopes::ATA::Config ataConfig;
    Telescopes::Generic::Config genericConfig;

    switch (config.telescope) {
        case TelescopeId::ATA:
            ataConfig.telescope = config.telescope;
            BL_CHECK(Telescopes::ATA::CollectUserInput(argc, argv, ataConfig));
            return Telescopes::ATA::Setup(ataConfig);
        case TelescopeId::GENERIC:
            genericConfig.telescope = config.telescope;
            BL_CHECK(Telescopes::Generic::CollectUserInput(argc, argv, genericConfig));
            return Telescopes::Generic::Setup(genericConfig);
        default:
            BL_FATAL("Telescope not implemented yet.");
    }

    return Result::ERROR;
}

const Result SetupProcessingPipeline(int argc, char **argv, const Config& config) {
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::milliseconds;

    auto t1 = high_resolution_clock::now();

    BL_CHECK((SetupTelescope(argc, argv, config)));

    auto t2 = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(t2 - t1).count();
    BL_INFO("Setup and processing finished in {} milliseconds!", duration);

    return Result::SUCCESS;
}

const Result CollectUserInput(int argc, char **argv, Config& config) {
    ::CLI::App app("BLADE (Breakthrough Listen Accelerated DSP Engine) - Command Line Tool");

    app.allow_extras();
    app.remove_option(app.get_help_ptr());

    // Read target telescope. 
    app
        .add_option("-t,--telescope", config.telescope, "Telescope ID (ATA, GENERIC)")
            ->required()
            ->transform(::CLI::CheckedTransformer(TelescopeMap, ::CLI::ignore_case));

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
    BL_CHECK(SetupProcessingPipeline(argc, argv, config));

    return Result::SUCCESS;
}

}  // namespace Blade::CLI
