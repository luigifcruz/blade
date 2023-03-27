#ifndef BLADE_LOGGER_HH
#define BLADE_LOGGER_HH

#include <iostream>

#include <fmt/ostream.h>
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/color.h>
#include <fmt/ranges.h>

#include "blade_config.hh"

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define BL_LOG_HEAD_DECR fmt::emphasis::bold
#define BL_LOG_HEAD_NAME fmt::format(BL_LOG_HEAD_DECR, "BLADE ")
#define BL_LOG_HEAD_FILE fmt::format(BL_LOG_HEAD_DECR, "[{}@{}] ", __FILENAME__, __LINE__)
#define BL_LOG_HEAD_TRACE fmt::format(BL_LOG_HEAD_DECR, "[TRACE] ")
#define BL_LOG_HEAD_DEBUG fmt::format(BL_LOG_HEAD_DECR, "[DEBUG] ")
#define BL_LOG_HEAD_WARN fmt::format(BL_LOG_HEAD_DECR, "[WARN]  ")
#define BL_LOG_HEAD_INFO fmt::format(BL_LOG_HEAD_DECR, "[INFO]  ")
#define BL_LOG_HEAD_ERROR fmt::format(BL_LOG_HEAD_DECR, "[ERROR] ")
#define BL_LOG_HEAD_FATAL fmt::format(BL_LOG_HEAD_DECR, "[FATAL] ")

#define BL_LOG_HEAD_SEPR fmt::format(BL_LOG_HEAD_DECR, "| ")

#ifndef BL_DISABLE_PRINT
#define BL_DISABLE_PRINT() std::cout.setstate(std::ios_base::failbit);
#endif

#ifndef BL_ENABLE_PRINT
#define BL_ENABLE_PRINT() std::cout.clear();
#endif

#if defined(BL_LOG_DOMAIN)
#define BL_LOG_DOMAIN_STR fmt::format(BL_LOG_HEAD_DECR, "[{}] ", BL_LOG_DOMAIN)
#else
#define BL_LOG_DOMAIN_STR ""
#endif

#ifndef BL_TRACE
#ifndef NDEBUG
#define BL_TRACE(...) if (getenv("TRACE")) std::cout << BL_LOG_HEAD_NAME << BL_LOG_HEAD_FILE << BL_LOG_HEAD_TRACE << BL_LOG_DOMAIN_STR << \
        BL_LOG_HEAD_SEPR << fmt::format(fg(fmt::color::white), __VA_ARGS__) << std::endl;
#else
#define BL_TRACE(...)
#endif
#endif

#ifndef BL_DEBUG
#ifndef NDEBUG
#define BL_DEBUG(...) std::cout << BL_LOG_HEAD_NAME << BL_LOG_HEAD_DEBUG << BL_LOG_HEAD_SEPR << BL_LOG_DOMAIN_STR << \
        fmt::format(fg(fmt::color::orange), __VA_ARGS__) << std::endl;
#else
#define BL_DEBUG(...)
#endif
#endif

#ifndef BL_WARN
#define BL_WARN(...) std::cout << BL_LOG_HEAD_NAME << BL_LOG_HEAD_WARN << BL_LOG_HEAD_SEPR << BL_LOG_DOMAIN_STR << \
        fmt::format(fg(fmt::color::yellow), __VA_ARGS__) << std::endl;
#else
#define BL_WARN(...)
#endif

#ifndef BL_INFO
#define BL_INFO(...) std::cout << BL_LOG_HEAD_NAME << BL_LOG_HEAD_INFO << BL_LOG_HEAD_SEPR << BL_LOG_DOMAIN_STR << \
        fmt::format(fg(fmt::color::cyan), __VA_ARGS__) << std::endl;
#else
#define BL_INFO(...)
#endif

#ifndef BL_ERROR
#define BL_ERROR(...) std::cerr << BL_LOG_HEAD_NAME << BL_LOG_HEAD_FILE << BL_LOG_HEAD_ERROR << BL_LOG_DOMAIN_STR << \
        BL_LOG_HEAD_SEPR << fmt::format(fg(fmt::color::red), __VA_ARGS__) << std::endl;
#else
#define BL_ERROR(...)
#endif

#ifndef BL_FATAL
#define BL_FATAL(...) std::cerr << BL_LOG_HEAD_NAME << BL_LOG_HEAD_FILE << BL_LOG_HEAD_FATAL << BL_LOG_DOMAIN_STR << \
        BL_LOG_HEAD_SEPR << fmt::format(fg(fmt::color::magenta), __VA_ARGS__) << std::endl;
#else
#define BL_FATAL(...)
#endif

inline void BL_LOG_PRINT_ET() {
    BL_INFO(R"(

Welcome to BLADE (Breakthrough Listen Accelerated DSP Engine)!
Version {} | Build Type: {} | Commit: {}
                   .-.
    .-""`""-.    |(0 0)
 _/`oOoOoOoOo`\_ \ \-/
'.-=-=-=-=-=-=-.' \/ \
  `-=.=-.-=.=-'    \ /\
     ^  ^  ^       _H_ \ art by jgs
    )", BLADE_VERSION_STR, BLADE_BUILD_TYPE, BLADE_COMMIT_STR);
}

#endif
