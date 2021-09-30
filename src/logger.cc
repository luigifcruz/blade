#include <spdlog/sinks/stdout_color_sinks.h>

#include "bl-beamformer/logger.hh"

std::string computeMethodName(const std::string& function, const std::string& prettyFunction) {
    size_t locFunName = prettyFunction.find(function);
    size_t begin = prettyFunction.rfind(" ",locFunName) + 1;
    size_t end = prettyFunction.find("(",locFunName + function.length());
    if (prettyFunction[end + 1] == ')')
        return (prettyFunction.substr(begin,end - begin) + "()");
    else
        return (prettyFunction.substr(begin,end - begin) + "(...)");
}

namespace BL {

Result Logger::Init() {
#if not defined(NDEBUG)
    // ASCII art made by jgs https://www.asciiart.eu/space/aliens
    std::cout << R"(
                       .-.
        .-""`""-.    |(0 0)
     _/`oOoOoOoOo`\_ \ \-/
    '.-=-=-=-=-=-=-.' \/ \
      `-=.=-.-=.=-'    \ /\
         ^  ^  ^       _H_ \
    )" << std::endl;
#endif

    auto consoleSink = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
    consoleSink->set_pattern("[%Y-%m-%d %T.%e%z] [%n] [%l] [%!] %v%$");

    std::vector<spdlog::sink_ptr> sinks{consoleSink};
    auto logger = std::make_shared<spdlog::logger>(BL_LOG_ID, sinks.begin(), sinks.end());

#ifdef NDEBUG
    logger->set_level(spdlog::level::warn);
    logger->flush_on(spdlog::level::warn);
#else
    logger->set_level(spdlog::level::trace);
    logger->flush_on(spdlog::level::trace);
#endif

    spdlog::register_logger(logger);

    return Result::SUCCESS;
}

Result Logger::Shutdown() {
    spdlog::shutdown();
    return Result::SUCCESS;
}

} // namespace BL::Logger
