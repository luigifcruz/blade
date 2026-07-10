#ifndef BLADE_CORRELATOR_LAUNCH_PLAN_HH
#define BLADE_CORRELATOR_LAUNCH_PLAN_HH

#include <algorithm>
#include <cstdint>
#include <string>

#include <jetstream/fmt/format.h>

namespace Jetstream::Modules {

struct CorrelatorLaunchPlan {
    bool packed = false;
    uint64_t antennaStride = 0;
    uint64_t tileGrid = 0;
    uint64_t tileCount = 0;
    uint64_t threadCount = 0;
    uint64_t tileBatchCount = 0;
    uint64_t chunkCount = 0;
    uint64_t chunkSamples = 0;
    uint64_t stageSamples = 0;
};

constexpr uint64_t kMaxIntegerSamples = 65535;

constexpr uint64_t kSharedMemoryBudget = 49152;

inline bool DeriveCorrelatorLaunchPlan(const uint64_t antennas,
                                       const uint64_t channels,
                                       const uint64_t samples,
                                       const bool integerInput,
                                       const bool integerCalculation,
                                       const uint64_t calculationScalarSize,
                                       CorrelatorLaunchPlan& plan,
                                       std::string& error) {
    if (antennas == 0 || channels == 0 || samples == 0) {
        error = "the input dimensions must all be positive";
        return false;
    }

    plan.packed = integerInput && (samples % 4) == 0 && samples <= kMaxIntegerSamples;

    const uint64_t antennasEven = antennas + (antennas % 2);
    plan.antennaStride = antennasEven + 2;
    if ((plan.antennaStride % 4) != 2) {
        plan.antennaStride += 2;
    }

    plan.tileGrid = antennasEven / 2;
    plan.tileCount = (plan.tileGrid * (plan.tileGrid + 1)) / 2;

    plan.threadCount = std::clamp(((plan.tileCount + 31) / 32) * 32,
                                  static_cast<uint64_t>(64),
                                  static_cast<uint64_t>(256));
    plan.tileBatchCount = (plan.tileCount + plan.threadCount - 1) / plan.threadCount;

    constexpr uint64_t kTargetBlocks = 1024;
    constexpr uint64_t kMinChunkSamples = 128;
    constexpr uint64_t kMaxChunkCount = 256;

    const auto pickChunks = [&](const bool split) {
        plan.chunkCount = 1;
        if (split) {
            while (plan.chunkCount < kMaxChunkCount &&
                   (samples % (2 * plan.chunkCount)) == 0 &&
                   (samples / (2 * plan.chunkCount)) >= kMinChunkSamples &&
                   (channels * plan.chunkCount) < kTargetBlocks) {
                plan.chunkCount *= 2;
            }
        }
        plan.chunkSamples = samples / plan.chunkCount;
    };

    const auto chooseStage = [&](const uint64_t bytesPerSample, const uint64_t multiple) -> uint64_t {
        for (uint64_t stage = 128; stage >= 1; stage /= 2) {
            if ((stage % multiple) == 0 &&
                (stage * bytesPerSample) <= kSharedMemoryBudget &&
                (plan.chunkSamples % stage) == 0) {
                return stage;
            }
        }
        return 0;
    };

    plan.stageSamples = 0;
    if (plan.packed) {
        pickChunks(true);
        plan.stageSamples = chooseStage(plan.antennaStride * 4, 4);
        if (plan.stageSamples == 0) {
            plan.packed = false;
        }
    }

    if (!plan.packed) {
        pickChunks(false);
        plan.stageSamples = chooseStage(4 * plan.antennaStride * calculationScalarSize, 1);
    }

    if (plan.stageSamples == 0) {
        error = "the input does not fit the shared-memory budget";
        return false;
    }

    if (!plan.packed && integerCalculation && plan.chunkSamples > kMaxIntegerSamples) {
        error = "the time dimension overflows a 32-bit integer accumulator, "
                "use a floating-point calculation mode";
        return false;
    }

    return true;
}

inline std::string BuildCorrelatorConstants(const CorrelatorLaunchPlan& plan,
                                            const uint64_t antennas,
                                            const uint64_t channels,
                                            const uint64_t samples,
                                            const uint64_t conjugateAntennaIndex) {
    const std::string packedConstants = plan.packed
                                            ? jst::fmt::format(
                                                  "static constexpr U64 STAGE_GROUPS = {}ull;\n",
                                                  plan.stageSamples / 4)
                                            : "";

    return jst::fmt::format(
        "static constexpr U64 A = {}ull;\n"
        "static constexpr U64 C = {}ull;\n"
        "static constexpr U64 T = {}ull;\n"
        "static constexpr U64 TILE_GRID = {}ull;\n"
        "static constexpr U64 TILE_COUNT = {}ull;\n"
        "static constexpr U64 ANTENNA_STRIDE = {}ull;\n"
        "static constexpr int THREADS = {};\n"
        "static constexpr U64 STAGE_SAMPLES = {}ull;\n"
        "static constexpr U64 CHUNK_SAMPLES = {}ull;\n"
        "static constexpr U64 STAGE_COUNT = {}ull;\n"
        "static constexpr U64 CONJUGATE_ANTENNA = {}ull;\n"
        "{}",
        antennas,
        channels,
        samples,
        plan.tileGrid,
        plan.tileCount,
        plan.antennaStride,
        plan.threadCount,
        plan.stageSamples,
        plan.chunkSamples,
        plan.chunkSamples / plan.stageSamples,
        conjugateAntennaIndex,
        packedConstants);
}

inline std::string BuildCorrelatorTypeAliases(const std::string& inputScalarType,
                                              const std::string& calculationScalarType) {
    return jst::fmt::format("using U64 = unsigned long long;\n"
                            "using U32 = unsigned int;\n"
                            "using I32 = int;\n"
                            "using InputScalar = {};\n"
                            "using CalcScalar = {};",
                            inputScalarType,
                            calculationScalarType);
}

}

#endif
