#ifndef BLADE_LOGGER_H
#define BLADE_LOGGER_H

#include <string>

#include "blade/common.hh"

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

#define BL_LOG_ID "BLADE"

#define BL_TRACE(...) if (spdlog::get(BL_LOG_ID)) \
{SPDLOG_LOGGER_TRACE(spdlog::get(BL_LOG_ID), __VA_ARGS__);}

#define BL_DEBUG(...) if (spdlog::get(BL_LOG_ID)) \
{SPDLOG_LOGGER_DEBUG(spdlog::get(BL_LOG_ID), __VA_ARGS__);}

#define BL_WARN(...)  if (spdlog::get(BL_LOG_ID)) \
{SPDLOG_LOGGER_WARN(spdlog::get(BL_LOG_ID), __VA_ARGS__);}

#define BL_INFO(...)  if (spdlog::get(BL_LOG_ID)) \
{SPDLOG_LOGGER_INFO(spdlog::get(BL_LOG_ID), __VA_ARGS__);}

#define BL_ERROR(...) if (spdlog::get(BL_LOG_ID)) \
{SPDLOG_LOGGER_ERROR(spdlog::get(BL_LOG_ID), __VA_ARGS__);}

#define BL_FATAL(...) if (spdlog::get(BL_LOG_ID)) \
{SPDLOG_LOGGER_CRITICAL(spdlog::get(BL_LOG_ID), __VA_ARGS__);}

namespace Blade {

class BLADE_API Logger {
 public:
    Logger();
    ~Logger();
};

}  // namespace Blade

#endif
