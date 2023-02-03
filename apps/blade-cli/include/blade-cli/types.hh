#ifndef BLADE_CLI_TYPES_HH
#define BLADE_CLI_TYPES_HH

#include "blade/base.hh"
#include "blade/types.hh"

namespace Blade::CLI {

enum class TelescopeId : uint8_t {
    ATA,
    GENERIC,
};

inline std::unordered_map<std::string, TelescopeId> TelescopeMap = {
    {"ATA",     TelescopeId::ATA},
    {"GENERIC", TelescopeId::GENERIC},
};

typedef struct {
    TelescopeId telescope;
} Config;

}  // namespace Blade::CLI

#endif
