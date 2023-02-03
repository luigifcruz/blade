#ifndef BLADE_CLI_TELESCOPES_GENERIC_CONFIG_HH
#define BLADE_CLI_TELESCOPES_GENERIC_CONFIG_HH

namespace Blade::CLI::Telescopes::Generic {

enum class TypeId : uint8_t {
    CI8, 
    CF16,
    CF32,
    CF64,
    I8,
    F16,
    F32,
    F64,
};

inline std::unordered_map<std::string, TypeId> TypeMap = {
    {"CI8",      TypeId::CI8}, 
    {"CF16",     TypeId::CF16}, 
    {"CF32",     TypeId::CF32}, 
    {"CF64",     TypeId::CF64}, 
    {"I8",       TypeId::I8}, 
    {"F16",      TypeId::F16}, 
    {"F32",      TypeId::F32}, 
    {"F64",      TypeId::F64}, 
};

enum class ModeId : uint8_t {
    MODE_H,
};

inline std::unordered_map<std::string, ModeId> ModeMap = {
    {"MODE_H",     ModeId::MODE_H},
    {"H",          ModeId::MODE_H},
};

typedef struct {
    ModeId mode;
    TelescopeId telescope;
    std::string inputGuppiFile;
    std::string outputFile;
    TypeId inputType;
    TypeId outputType;
    U64 numberOfWorkers;
    U64 channelizerRate;
} Config;

}  // namespace Blade::CLI::Telecopes::Generic

#endif // BLADE_CLI_TELESCOPES_GENERIC_CONFIG_HH