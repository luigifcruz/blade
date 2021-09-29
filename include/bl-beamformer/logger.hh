#ifndef BL_LOGGER_H
#define BL_LOGGER_H

#include "bl-beamformer/type.hh"

#undef SPDLOG_FUNCTION
std::string computeMethodName(const std::string& function, const std::string& prettyFunction);
#define SPDLOG_FUNCTION computeMethodName(__FUNCTION__, __PRETTY_FUNCTION__).c_str()
#include <spdlog/spdlog.h>

#define BL_LOG_ID "BL"

#define BL_TRACE(...) if (spdlog::get(BL_LOG_ID)) {SPDLOG_LOGGER_TRACE(spdlog::get(BL_LOG_ID), __VA_ARGS__);}
#define BL_DEBUG(...) if (spdlog::get(BL_LOG_ID)) {SPDLOG_LOGGER_DEBUG(spdlog::get(BL_LOG_ID), __VA_ARGS__);}
#define BL_WARN(...)  if (spdlog::get(BL_LOG_ID)) {SPDLOG_LOGGER_WARN(spdlog::get(BL_LOG_ID), __VA_ARGS__);}
#define BL_INFO(...)  if (spdlog::get(BL_LOG_ID)) {SPDLOG_LOGGER_INFO(spdlog::get(BL_LOG_ID), __VA_ARGS__);}
#define BL_ERROR(...) if (spdlog::get(BL_LOG_ID)) {SPDLOG_LOGGER_ERROR(spdlog::get(BL_LOG_ID), __VA_ARGS__);}
#define BL_FATAL(...) if (spdlog::get(BL_LOG_ID)) {SPDLOG_LOGGER_CRITICAL(spdlog::get(BL_LOG_ID), __VA_ARGS__);}

namespace BL {

    class Logger {
    public:
        Logger() = default;
        ~Logger() = default;

        static Result Init();
        static Result Shutdown();
    };

}

#endif
