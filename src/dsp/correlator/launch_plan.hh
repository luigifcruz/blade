#ifndef BLADE_CORRELATOR_LAUNCH_PLAN_HH
#define BLADE_CORRELATOR_LAUNCH_PLAN_HH

#include <algorithm>
#include <cstdint>
#include <string>

#include <jetstream/fmt/format.h>
#include <jetstream/tools/numeric.hh>

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
    plan = {};
    error.clear();

    if (antennas == 0 || channels == 0 || samples == 0) {
        error = "the input dimensions must all be positive";
        return false;
    }

    plan.packed = integerInput && (samples % 4) == 0 && samples <= kMaxIntegerSamples;

    uint64_t antennasEven = 0;
    if (!detail::CheckedAdd(antennas, antennas % 2, antennasEven) ||
        !detail::CheckedAdd(antennasEven, uint64_t{2}, plan.antennaStride)) {
        error = "the antenna geometry exceeds the supported range";
        return false;
    }
    if ((plan.antennaStride % 4) != 2) {
        if (!detail::CheckedAdd(plan.antennaStride,
                                uint64_t{2},
                                plan.antennaStride)) {
            error = "the antenna stride exceeds the supported range";
            return false;
        }
    }

    plan.tileGrid = antennasEven / 2;
    uint64_t tileGridPlusOne = 0;
    uint64_t tileProduct = 0;
    if (!detail::CheckedAdd(plan.tileGrid,
                            uint64_t{1},
                            tileGridPlusOne) ||
        !detail::CheckedMultiply(plan.tileGrid,
                                 tileGridPlusOne,
                                 tileProduct)) {
        error = "the baseline tile count exceeds the supported range";
        return false;
    }
    plan.tileCount = tileProduct / 2;

    if (plan.tileCount >= 256) {
        plan.threadCount = 256;
    } else {
        const uint64_t warpCount = (plan.tileCount / 32) +
                                   (plan.tileCount % 32 != 0);
        plan.threadCount = std::clamp(warpCount * 32,
                                      uint64_t{64},
                                      uint64_t{256});
    }
    plan.tileBatchCount = (plan.tileCount / plan.threadCount) +
                          (plan.tileCount % plan.threadCount != 0);

    constexpr uint64_t kTargetBlocks = 1024;
    constexpr uint64_t kMinChunkSamples = 128;
    constexpr uint64_t kMaxChunkCount = 256;

    const auto pickChunks = [&](const bool split) {
        plan.chunkCount = 1;
        if (split) {
            while (plan.chunkCount < kMaxChunkCount) {
                uint64_t nextChunkCount = 0;
                uint64_t activeBlockCount = 0;
                if (!detail::CheckedMultiply(plan.chunkCount,
                                             uint64_t{2},
                                             nextChunkCount) ||
                    !detail::CheckedMultiply(channels,
                                             plan.chunkCount,
                                             activeBlockCount) ||
                    (samples % nextChunkCount) != 0 ||
                    (samples / nextChunkCount) < kMinChunkSamples ||
                    activeBlockCount >= kTargetBlocks) {
                    break;
                }
                plan.chunkCount = nextChunkCount;
            }
        }
        plan.chunkSamples = samples / plan.chunkCount;
    };

    const auto chooseStage = [&](const uint64_t bytesPerSample, const uint64_t multiple) -> uint64_t {
        for (uint64_t stage = 128; stage >= 1; stage /= 2) {
            uint64_t sharedMemorySize = 0;
            if ((stage % multiple) == 0 &&
                detail::CheckedMultiply(stage,
                                        bytesPerSample,
                                        sharedMemorySize) &&
                sharedMemorySize <= kSharedMemoryBudget &&
                (plan.chunkSamples % stage) == 0) {
                return stage;
            }
        }
        return 0;
    };

    plan.stageSamples = 0;
    if (plan.packed) {
        pickChunks(true);
        uint64_t bytesPerSample = 0;
        if (detail::CheckedMultiply(plan.antennaStride,
                                    uint64_t{4},
                                    bytesPerSample)) {
            plan.stageSamples = chooseStage(bytesPerSample, 4);
        }
        if (plan.stageSamples == 0) {
            plan.packed = false;
        }
    }

    if (!plan.packed) {
        pickChunks(false);
        uint64_t bytesPerSample = 0;
        if (detail::CheckedMultiply(plan.antennaStride,
                                    uint64_t{4},
                                    bytesPerSample) &&
            detail::CheckedMultiply(bytesPerSample,
                                    calculationScalarSize,
                                    bytesPerSample)) {
            plan.stageSamples = chooseStage(bytesPerSample, 1);
        }
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
