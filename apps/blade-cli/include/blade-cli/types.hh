#ifndef BLADE_CLI_TYPES_HH
#define BLADE_CLI_TYPES_HH

#include "blade/base.hh"
#include "blade/types.hh"

namespace Blade::CLI {

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

enum class TelescopeId : uint8_t {
    ATA,
    VLA,
    MEERKAT,
    GENERIC,
};

inline std::unordered_map<std::string, TelescopeId> TelescopeMap = {
    {"ATA",     TelescopeId::ATA}, 
    {"VLA",     TelescopeId::VLA},
    {"MEERKAT", TelescopeId::MEERKAT},
    {"GENERIC", TelescopeId::GENERIC},
};

enum class ModeId : uint8_t {
    MODE_B,
    MODE_H,
    MODE_BH,
    MODE_BS,
};

inline std::unordered_map<std::string, ModeId> ModeMap = {
    {"MODE_B",     ModeId::MODE_B}, 
    {"MODE_BS",    ModeId::MODE_BS}, 
    {"MODE_H",     ModeId::MODE_H}, 
    {"MODE_BH",    ModeId::MODE_BH}, 
    {"B",          ModeId::MODE_B}, 
    {"BS",         ModeId::MODE_BS}, 
    {"H",          ModeId::MODE_H}, 
    {"BH",         ModeId::MODE_BH}, 
};

typedef struct {
    ModeId mode;
    TelescopeId telescope;
    std::string inputGuppiFile;
    std::string inputBfr5File;
    std::string outputFile;
    TypeId inputType;
    TypeId outputType;
    U64 numberOfWorkers;
    U64 preBeamformerChannelizerRate;
    U64 stepNumberOfTimeSamples;
    U64 stepNumberOfFrequencyChannels;
} Config;

}  // namespace Blade::CLI

#endif
