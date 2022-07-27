#ifndef BLADE_CLI_TYPES
#define BLADE_CLI_TYPES

#include "blade/base.hh"

#include "blade/pipelines/generic/file_reader.hh"

using namespace Blade;

typedef enum {
    ATA,
    VLA,
    MEERKAT,
} TelescopeId;

inline std::map<std::string, TelescopeId> TelescopeMap = {
    {"ATA",     TelescopeId::ATA}, 
    {"VLA",     TelescopeId::VLA},
    {"MEERKAT", TelescopeId::MEERKAT}
};

typedef enum {
    MODE_B,
    MODE_A,
} ModeId;

inline std::map<std::string, ModeId> ModeMap = {
    {"MODE_B",     ModeId::MODE_B}, 
    {"MODE_A",     ModeId::MODE_A},
    {"B",          ModeId::MODE_B}, 
    {"A",          ModeId::MODE_A},
};

typedef struct {
    ModeId mode;
    TelescopeId telescope;
    std::string inputGuppiFile;
    std::string inputBfr5File;
    U64 numberOfWorkers;
    U64 preBeamformerChannelizerRate;
    U64 stepNumberOfTimeSamples;
    U64 stepNumberOfFrequencyChannels;
} CliConfig;

#endif
