#ifndef BL_LOGGER_H
#define BL_LOGGER_H

#include "bl-beamformer/type.hh"

#undef SPDLOG_FUNCTION
#undef SPDLOG_ACTIVE_LEVEL

#ifdef NDEBUG
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_WARN
#else
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
#endif

inline std::string computeMethodName(const std::string& function, const std::string& prettyFunction) {
    size_t locFunName = prettyFunction.find(function);
    size_t begin = prettyFunction.rfind(" ",locFunName) + 1;
    size_t end = prettyFunction.find("(",locFunName + function.length());
    if (prettyFunction[end + 1] == ')')
        return (prettyFunction.substr(begin,end - begin) + "()");
    else
        return (prettyFunction.substr(begin,end - begin) + "(...)");
}
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

    class BL_API Logger {
    public:
        Logger() = default;
        ~Logger() = default;

        static Result Init();
        static Result Shutdown();
    };

}

#endif
