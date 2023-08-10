#ifndef BLADE_CLI_TELESCOPES_ATA_CONFIG_HH
#define BLADE_CLI_TELESCOPES_ATA_CONFIG_HH

namespace Blade::CLI::Telescopes::ATA {

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
    MODE_B,
    MODE_BS,
};

inline std::unordered_map<std::string, ModeId> ModeMap = {
    {"MODE_B",     ModeId::MODE_B},
    {"MODE_BS",    ModeId::MODE_BS},
    {"B",          ModeId::MODE_B},
    {"BS",         ModeId::MODE_BS},
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
    U64 integrationSize;
    F64 snrThreshold;
    F64 driftRateMinimum;
    F64 driftRateMaximum;
    BOOL driftRateZeroExcluded;
    BOOL incoherentBeamEnabled;
    BOOL progressBarDisabled;
    I64 hitsGroupingMargin;
    BOOL produceDebugHits;
    F64 stampFrequencyMarginHz;
    BOOL phasorNegateDelays;
    U64 inputGuppiFileLimit;
} Config;

}  // namespace Blade::CLI::Telecopes::ATA

#endif // BLADE_CLI_TELESCOPES_ATA_CONFIG_HH