#ifndef BLADE_LOGGER_HH
#define BLADE_LOGGER_HH

#include <string>

#include "blade/macros.hh"

#undef SPDLOG_ACTIVE_LEVEL
#ifdef NDEBUG
    #define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_WARN
#else
    #define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
#endif

#undef  SPDLOG_FUNCTION
BLADE_API std::string computeMethodName(const std::string&, const std::string&);
#define SPDLOG_FUNCTION computeMethodName(__FUNCTION__, __PRETTY_FUNCTION__).c_str()

#include <spdlog/spdlog.h>
#include <fmt/core.h>

#define BL_LOG_ID "BLADE"

#ifndef BL_TRACE
#define BL_TRACE(...) if (spdlog::get(BL_LOG_ID)) \
{fprintf(stderr, "%s\n", fmt::format(__VA_ARGS__).c_str());}
#endif

#ifndef BL_DEBUG
#define BL_DEBUG(...) if (spdlog::get(BL_LOG_ID)) \
{fprintf(stderr, "%s\n", fmt::format(__VA_ARGS__).c_str());}
#endif

#ifndef BL_WARN
#define BL_WARN(...)  if (spdlog::get(BL_LOG_ID)) \
{fprintf(stderr, "%s\n", fmt::format(__VA_ARGS__).c_str());}
#endif

#ifndef BL_INFO
#define BL_INFO(...)  if (spdlog::get(BL_LOG_ID)) \
{fprintf(stderr, "%s\n", fmt::format(__VA_ARGS__).c_str());}
#endif

#ifndef BL_ERROR
#define BL_ERROR(...) if (spdlog::get(BL_LOG_ID)) \
{fprintf(stderr, "%s\n", fmt::format(__VA_ARGS__).c_str());}
#endif

#ifndef BL_FATAL
#define BL_FATAL(...) if (spdlog::get(BL_LOG_ID)) \
{fprintf(stderr, "%s\n", fmt::format(__VA_ARGS__).c_str());}
#endif

namespace Blade {

class BLADE_API Logger {
 public:
    Logger();
    ~Logger();
};

}  // namespace Blade

#endif
