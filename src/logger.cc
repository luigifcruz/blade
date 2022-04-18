#include <spdlog/sinks/stdout_color_sinks.h>

#include "blade/logger.hh"

std::string computeMethodName(const std::string& function,
                              const std::string& prettyFunction) {
    std::size_t locFunName = prettyFunction.find(function);
    std::size_t begin = prettyFunction.rfind(" ", locFunName) + 1;
    std::size_t end = prettyFunction.find("(", locFunName + function.length());
    if (prettyFunction[end + 1] == ')')
        return (prettyFunction.substr(begin, end - begin) + "()");
    else
        return (prettyFunction.substr(begin, end - begin) + "(...)");
}

namespace Blade {

Logger::Logger() {
    if (spdlog::get(BL_LOG_ID)) {
        return;
    } 

    auto consoleSink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    consoleSink->set_pattern("[%Y-%m-%d %T.%e%z] [%n] [%l] [%!] %v%$");

    std::vector<spdlog::sink_ptr> sinks{consoleSink};
    auto logger = std::make_shared<spdlog::logger>(BL_LOG_ID,
            sinks.begin(), sinks.end());

    // If you are going to change this behavior,
    // remind to reflect the changes in the header file.
#ifdef NDEBUG
    logger->set_level(spdlog::level::warn);
    logger->flush_on(spdlog::level::warn);
#else
    logger->set_level(spdlog::level::trace);
    logger->flush_on(spdlog::level::trace);
#endif
    //

    spdlog::register_logger(logger);

    logger->info(R"(

Welcome to BLADE (Breakthrough Listen Accelerated DSP Engine)!
                   .-.
    .-""`""-.    |(0 0)
 _/`oOoOoOoOo`\_ \ \-/
'.-=-=-=-=-=-=-.' \/ \
  `-=.=-.-=.=-'    \ /\
     ^  ^  ^       _H_ \
    )");
}

Logger::~Logger() {
    spdlog::shutdown();
}

}  // namespace Blade
