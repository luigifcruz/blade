#include <algorithm>

#include <jetstream/backend/devices/cuda/helpers.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>
#include <jetstream/runtime_context_native_cuda.hh>
#include <jetstream/scheduler_context.hh>

#include "launch_plan.hh"
#include "module_impl.hh"

namespace Jetstream::Modules {

namespace {

constexpr const char* kCorrelatorKernelName = "correlator";

// Both kernels share the same decomposition. A thread block owns one channel and
// one contiguous slice of the time axis. It stages every antenna's voltages for
// that slice into shared memory, then each thread reduces one 2x2 tile of the
// upper-triangular baseline matrix over the whole slice while holding the four
// polarization products of its four baselines in registers. Input is therefore
// read from global memory exactly once per launch instead of once per baseline,
// and the only atomics are the one flush per baseline at the end of the block.
//
// The antenna axis is padded to an even count so it always tiles by two; the
// padding columns stay zeroed and their baselines are never written out.

constexpr const char* kCorrelatorPackedKernelSource = R"(
<<<type_aliases>>>
<<<kernel_constants>>>

struct alignas(16) PackedQuad {
    U32 x, y, z, w;
};

struct alignas(8) PackedPair {
    U32 x, y;
};

__device__ __forceinline__ U32 permuteBytes(U32 lhs, U32 rhs, U32 selector) {
    U32 result;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(result) : "r"(lhs), "r"(rhs), "r"(selector));
    return result;
}

__device__ __forceinline__ I32 dotProductAccumulate(U32 lhs, U32 rhs, I32 accumulator) {
    I32 result;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(result) : "r"(lhs), "r"(rhs), "r"(accumulator));
    return result;
}

// Input is streamed once and never revisited, so keep it out of the way of the
// output visibilities in L2.
__device__ __forceinline__ PackedQuad loadStreaming(const PackedQuad* address) {
    PackedQuad quad;
    asm("ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];"
        : "=r"(quad.x), "=r"(quad.y), "=r"(quad.z), "=r"(quad.w)
        : "l"(address));
    return quad;
}

// One antenna's four consecutive time samples arrive interleaved as
// (Xr Xi Yr Yi) per sample. dp4a wants the opposite: one component packed across
// four samples. Transpose the 4x4 byte matrix with four PRMT pairs.
__device__ __forceinline__ void deinterleave(const PackedQuad& quad,
                                             U32& xReal, U32& xImag,
                                             U32& yReal, U32& yImag) {
    xReal = permuteBytes(permuteBytes(quad.x, quad.y, 0x0040), permuteBytes(quad.z, quad.w, 0x0040), 0x5410);
    xImag = permuteBytes(permuteBytes(quad.x, quad.y, 0x0051), permuteBytes(quad.z, quad.w, 0x0051), 0x5410);
    yReal = permuteBytes(permuteBytes(quad.x, quad.y, 0x0062), permuteBytes(quad.z, quad.w, 0x0062), 0x5410);
    yImag = permuteBytes(permuteBytes(quad.x, quad.y, 0x0073), permuteBytes(quad.z, quad.w, 0x0073), 0x5410);
}

// Flushes into an exact int32 accumulator, not the float output. Integer addition
// is associative, so the visibilities do not depend on the order in which the
// time-chunk blocks happen to land, which float atomics could not guarantee.
extern "C" __global__ __launch_bounds__(THREADS) void correlator(const PackedQuad* input,
                                                                 int* output) {
    __shared__ alignas(8) U32 plane[4][STAGE_GROUPS][ANTENNA_STRIDE];

    const U64 channel = blockIdx.x;
    const U64 timeBlock = blockIdx.y;
    const U64 threadIndex = threadIdx.x;
    const U64 tile = (static_cast<U64>(blockIdx.z) * THREADS) + threadIndex;
    const bool active = tile < TILE_COUNT;

    U64 tileRow = 0;
    U64 tileColumn = 0;
    if (active) {
        U64 remainder = tile;
        for (U64 row = 0; row < TILE_GRID; row++) {
            const U64 length = TILE_GRID - row;
            if (remainder < length) {
                tileRow = row;
                tileColumn = row + remainder;
                break;
            }
            remainder -= length;
        }
    }

    for (U64 index = threadIndex; index < 4 * STAGE_GROUPS * ANTENNA_STRIDE; index += THREADS) {
        (&plane[0][0][0])[index] = 0;
    }

    // Staging coordinates do not depend on the stage. Clamping the out-of-range
    // slots instead of skipping them keeps every load unconditional, so the
    // whole burst issues before the first unpack stalls on it.
    constexpr U64 STAGE_LOADS = ((A * STAGE_GROUPS) + THREADS - 1) / THREADS;
    U64 loadAntenna[STAGE_LOADS];
    U64 loadGroup[STAGE_LOADS];
    bool loadValid[STAGE_LOADS];
    #pragma unroll
    for (U64 slot = 0; slot < STAGE_LOADS; slot++) {
        const U64 index = threadIndex + (slot * THREADS);
        loadValid[slot] = index < (A * STAGE_GROUPS);
        loadAntenna[slot] = loadValid[slot] ? (index / STAGE_GROUPS) : 0;
        loadGroup[slot] = loadValid[slot] ? (index % STAGE_GROUPS) : 0;
    }

    // The imaginary part needs a subtraction that dp4a cannot express, so the
    // positive and negative halves accumulate separately and merge at the flush.
    // Both stay exact in 32-bit: CHUNK_SAMPLES is capped so they cannot overflow.
    I32 sumReal[2][2][4] = {};
    I32 sumImagPos[2][2][4] = {};
    I32 sumImagNeg[2][2][4] = {};

    const U64 timeBase = timeBlock * CHUNK_SAMPLES;

    for (U64 stage = 0; stage < STAGE_COUNT; stage++) {
        const U64 timeOrigin = timeBase + (stage * STAGE_SAMPLES);

        __syncthreads();

        PackedQuad staged[STAGE_LOADS];
        #pragma unroll
        for (U64 slot = 0; slot < STAGE_LOADS; slot++) {
            staged[slot] = loadStreaming(input + ((((loadAntenna[slot] * C) + channel) * T) + timeOrigin) / 4 +
                                         loadGroup[slot]);
        }
        #pragma unroll
        for (U64 slot = 0; slot < STAGE_LOADS; slot++) {
            if (!loadValid[slot]) {
                continue;
            }
            U32 xReal, xImag, yReal, yImag;
            deinterleave(staged[slot], xReal, xImag, yReal, yImag);
            plane[0][loadGroup[slot]][loadAntenna[slot]] = xReal;
            plane[1][loadGroup[slot]][loadAntenna[slot]] = xImag;
            plane[2][loadGroup[slot]][loadAntenna[slot]] = yReal;
            plane[3][loadGroup[slot]][loadAntenna[slot]] = yImag;
        }

        __syncthreads();

        if (!active) {
            continue;
        }

        #pragma unroll 4
        for (U64 group = 0; group < STAGE_GROUPS; group++) {
            PackedPair rowPlane[4];
            PackedPair columnPlane[4];
            #pragma unroll
            for (U64 component = 0; component < 4; component++) {
                rowPlane[component] = *reinterpret_cast<const PackedPair*>(&plane[component][group][2 * tileRow]);
                columnPlane[component] = *reinterpret_cast<const PackedPair*>(&plane[component][group][2 * tileColumn]);
            }

            #pragma unroll
            for (U64 row = 0; row < 2; row++) {
                const U32 axr = row ? rowPlane[0].y : rowPlane[0].x;
                const U32 axi = row ? rowPlane[1].y : rowPlane[1].x;
                const U32 ayr = row ? rowPlane[2].y : rowPlane[2].x;
                const U32 ayi = row ? rowPlane[3].y : rowPlane[3].x;

                #pragma unroll
                for (U64 column = 0; column < 2; column++) {
                    const U32 bxr = column ? columnPlane[0].y : columnPlane[0].x;
                    const U32 bxi = column ? columnPlane[1].y : columnPlane[1].x;
                    const U32 byr = column ? columnPlane[2].y : columnPlane[2].x;
                    const U32 byi = column ? columnPlane[3].y : columnPlane[3].x;

                    sumReal[row][column][0] = dotProductAccumulate(axr, bxr, dotProductAccumulate(axi, bxi, sumReal[row][column][0]));
                    sumImagPos[row][column][0] = dotProductAccumulate(axi, bxr, sumImagPos[row][column][0]);
                    sumImagNeg[row][column][0] = dotProductAccumulate(axr, bxi, sumImagNeg[row][column][0]);

                    sumReal[row][column][1] = dotProductAccumulate(axr, byr, dotProductAccumulate(axi, byi, sumReal[row][column][1]));
                    sumImagPos[row][column][1] = dotProductAccumulate(axi, byr, sumImagPos[row][column][1]);
                    sumImagNeg[row][column][1] = dotProductAccumulate(axr, byi, sumImagNeg[row][column][1]);

                    sumReal[row][column][2] = dotProductAccumulate(ayr, bxr, dotProductAccumulate(ayi, bxi, sumReal[row][column][2]));
                    sumImagPos[row][column][2] = dotProductAccumulate(ayi, bxr, sumImagPos[row][column][2]);
                    sumImagNeg[row][column][2] = dotProductAccumulate(ayr, bxi, sumImagNeg[row][column][2]);

                    sumReal[row][column][3] = dotProductAccumulate(ayr, byr, dotProductAccumulate(ayi, byi, sumReal[row][column][3]));
                    sumImagPos[row][column][3] = dotProductAccumulate(ayi, byr, sumImagPos[row][column][3]);
                    sumImagNeg[row][column][3] = dotProductAccumulate(ayr, byi, sumImagNeg[row][column][3]);
                }
            }
        }
    }

    if (!active) {
        return;
    }

    #pragma unroll
    for (U64 row = 0; row < 2; row++) {
        #pragma unroll
        for (U64 column = 0; column < 2; column++) {
            const U64 antennaA = (2 * tileRow) + row;
            const U64 antennaB = (2 * tileColumn) + column;

            if (antennaA >= A || antennaB >= A || antennaB < antennaA) {
                continue;
            }

            const U64 baseline = ((antennaA * (2 * A - antennaA + 1)) / 2) + (antennaB - antennaA);
            int* destination = output + ((baseline * C * 4) + (channel * 4)) * 2;

            #pragma unroll
            for (U64 product = 0; product < 4; product++) {
                const I32 real = sumReal[row][column][product];
                const I32 imag = (CONJUGATE_ANTENNA == 1)
                                     ? (sumImagPos[row][column][product] - sumImagNeg[row][column][product])
                                     : (sumImagNeg[row][column][product] - sumImagPos[row][column][product]);
                atomicAdd(destination + (2 * product) + 0, real);
                atomicAdd(destination + (2 * product) + 1, imag);
            }
        }
    }
}
)";

constexpr const char* kAccumulateKernelName = "accumulate";
constexpr const char* kAccumulateKernelSource = R"(
using U64 = unsigned long long;
<<<accumulate_constants>>>

extern "C" __global__ void accumulate(int* scratch, float* output) {
    const U64 index = (static_cast<U64>(blockIdx.x) * blockDim.x) + threadIdx.x;

    if (index >= ELEMENT_COUNT) {
        return;
    }

    output[index] += static_cast<float>(scratch[index]);
    scratch[index] = 0;
}
)";

constexpr const char* kCorrelatorGenericKernelSource = R"(
<<<type_aliases>>>
<<<kernel_constants>>>

struct alignas(4 * sizeof(InputScalar)) InputQuad {
    InputScalar xReal, xImag, yReal, yImag;
};

extern "C" __global__ __launch_bounds__(THREADS) void correlator(const InputScalar* input,
                                                                 float* output) {
    __shared__ CalcScalar plane[4][STAGE_SAMPLES][ANTENNA_STRIDE];

    const U64 channel = blockIdx.x;
    const U64 timeBlock = blockIdx.y;
    const U64 threadIndex = threadIdx.x;
    const U64 tile = (static_cast<U64>(blockIdx.z) * THREADS) + threadIndex;
    const bool active = tile < TILE_COUNT;

    U64 tileRow = 0;
    U64 tileColumn = 0;
    if (active) {
        U64 remainder = tile;
        for (U64 row = 0; row < TILE_GRID; row++) {
            const U64 length = TILE_GRID - row;
            if (remainder < length) {
                tileRow = row;
                tileColumn = row + remainder;
                break;
            }
            remainder -= length;
        }
    }

    for (U64 index = threadIndex; index < 4 * STAGE_SAMPLES * ANTENNA_STRIDE; index += THREADS) {
        (&plane[0][0][0])[index] = static_cast<CalcScalar>(0);
    }

    CalcScalar sumReal[2][2][4] = {};
    CalcScalar sumImag[2][2][4] = {};

    const U64 timeBase = timeBlock * CHUNK_SAMPLES;

    for (U64 stage = 0; stage < STAGE_COUNT; stage++) {
        const U64 timeOrigin = timeBase + (stage * STAGE_SAMPLES);

        __syncthreads();

        for (U64 index = threadIndex; index < (A * STAGE_SAMPLES); index += THREADS) {
            const U64 antenna = index / STAGE_SAMPLES;
            const U64 sample = index % STAGE_SAMPLES;
            const U64 offset = ((((antenna * C) + channel) * T) + timeOrigin) + sample;
            const InputQuad quad = *reinterpret_cast<const InputQuad*>(input + (offset * 4));
            plane[0][sample][antenna] = static_cast<CalcScalar>(quad.xReal);
            plane[1][sample][antenna] = static_cast<CalcScalar>(quad.xImag);
            plane[2][sample][antenna] = static_cast<CalcScalar>(quad.yReal);
            plane[3][sample][antenna] = static_cast<CalcScalar>(quad.yImag);
        }

        __syncthreads();

        if (!active) {
            continue;
        }

        // Sum each stage separately and fold the partial into the running total.
        // Without a time split across blocks there is nothing else to keep the
        // floating-point summation chain short, and a chain of T terms would lose
        // roughly sqrt(T) times more precision than one of STAGE_SAMPLES terms.
        CalcScalar stageReal[2][2][4] = {};
        CalcScalar stageImag[2][2][4] = {};

        #pragma unroll 4
        for (U64 sample = 0; sample < STAGE_SAMPLES; sample++) {
            #pragma unroll
            for (U64 row = 0; row < 2; row++) {
                const CalcScalar axr = plane[0][sample][(2 * tileRow) + row];
                const CalcScalar axi = plane[1][sample][(2 * tileRow) + row];
                const CalcScalar ayr = plane[2][sample][(2 * tileRow) + row];
                const CalcScalar ayi = plane[3][sample][(2 * tileRow) + row];

                #pragma unroll
                for (U64 column = 0; column < 2; column++) {
                    const CalcScalar bxr = plane[0][sample][(2 * tileColumn) + column];
                    const CalcScalar bxi = plane[1][sample][(2 * tileColumn) + column];
                    const CalcScalar byr = plane[2][sample][(2 * tileColumn) + column];
                    const CalcScalar byi = plane[3][sample][(2 * tileColumn) + column];

                    stageReal[row][column][0] += (axr * bxr) + (axi * bxi);
                    stageImag[row][column][0] += (axi * bxr) - (axr * bxi);

                    stageReal[row][column][1] += (axr * byr) + (axi * byi);
                    stageImag[row][column][1] += (axi * byr) - (axr * byi);

                    stageReal[row][column][2] += (ayr * bxr) + (ayi * bxi);
                    stageImag[row][column][2] += (ayi * bxr) - (ayr * bxi);

                    stageReal[row][column][3] += (ayr * byr) + (ayi * byi);
                    stageImag[row][column][3] += (ayi * byr) - (ayr * byi);
                }
            }
        }

        #pragma unroll
        for (U64 row = 0; row < 2; row++) {
            #pragma unroll
            for (U64 column = 0; column < 2; column++) {
                #pragma unroll
                for (U64 product = 0; product < 4; product++) {
                    sumReal[row][column][product] += stageReal[row][column][product];
                    sumImag[row][column][product] += stageImag[row][column][product];
                }
            }
        }
    }

    if (!active) {
        return;
    }

    #pragma unroll
    for (U64 row = 0; row < 2; row++) {
        #pragma unroll
        for (U64 column = 0; column < 2; column++) {
            const U64 antennaA = (2 * tileRow) + row;
            const U64 antennaB = (2 * tileColumn) + column;

            if (antennaA >= A || antennaB >= A || antennaB < antennaA) {
                continue;
            }

            const U64 baseline = ((antennaA * (2 * A - antennaA + 1)) / 2) + (antennaB - antennaA);
            float* destination = output + ((baseline * C * 4) + (channel * 4)) * 2;

            #pragma unroll
            for (U64 product = 0; product < 4; product++) {
                const float real = static_cast<float>(sumReal[row][column][product]);
                float imag = static_cast<float>(sumImag[row][column][product]);
                if constexpr (CONJUGATE_ANTENNA == 0) {
                    imag = -imag;
                }
                atomicAdd(destination + (2 * product) + 0, real);
                atomicAdd(destination + (2 * product) + 1, imag);
            }
        }
    }
}
)";

}  // namespace

struct CorrelatorImplNativeCuda : public CorrelatorImpl,
                                  public NativeCudaRuntimeContext,
                                  public Scheduler::Context {
 public:
    Result create() final;
    Result destroy() override;
    Result computeInitialize() override;
    Result computeSubmit(const cudaStream_t& stream) override;
    Result computeDeinitialize() override;

 private:
    Result derivePlan();

    bool kernelCreated = false;
    bool accumulateKernelCreated = false;
    bool accumulatorCleared = false;

    Tensor accumulatorTensor;

    CorrelatorLaunchPlan plan;
    std::string inputScalarType;
    std::string calculationScalarType;
    U64 elementCount = 0;
};

Result CorrelatorImplNativeCuda::create() {
    const Tensor& input = inputs().at("buffer").tensor;

    if (input.dtype() != DataType::CI8 && input.dtype() != DataType::CF32) {
        JST_ERROR("[MODULE_CORRELATOR_NATIVE_CUDA] Unsupported input data type '{}'. Expected CI8 or CF32.",
                  input.dtype());
        return Result::ERROR;
    }

    JST_CHECK(CorrelatorImpl::create());
    JST_CHECK(derivePlan());

    elementCount = baselineCount * inputTensor.shape(kFrequencyAxis) * kOutputPolarizations * 2;

    if (plan.packed) {
        JST_CHECK(accumulatorTensor.create(inputTensor.device(), DataType::CI32, outputTensor.shape()));
        accumulatorCleared = false;
    }

    return Result::SUCCESS;
}

Result CorrelatorImplNativeCuda::destroy() {
    accumulatorTensor = {};
    plan = {};
    elementCount = 0;
    accumulatorCleared = false;

    return CorrelatorImpl::destroy();
}

Result CorrelatorImplNativeCuda::derivePlan() {
    inputScalarType = [&]() -> std::string {
        switch (inputTensor.dtype()) {
            case DataType::CI8: return "signed char";
            case DataType::CF32: return "float";
            default: return "";
        }
    }();

    if (inputScalarType.empty()) {
        JST_ERROR("[MODULE_CORRELATOR_NATIVE_CUDA] Unsupported input data type '{}'. Expected CI8 or CF32.",
                  inputTensor.dtype());
        return Result::ERROR;
    }

    if (inputTensor.dtype() == DataType::CF32 && calculationMode == "integer") {
        JST_ERROR("[MODULE_CORRELATOR_NATIVE_CUDA] Integer calculation mode is not supported for CF32 input. "
                  "Use single_precision_fp or double_precision_fp.");
        return Result::ERROR;
    }

    calculationScalarType = [&]() -> std::string {
        if (calculationMode == "integer") {
            return "int";
        }

        if (calculationMode == "single_precision_fp") {
            return "float";
        }

        if (calculationMode == "double_precision_fp") {
            return "double";
        }

        return "";
    }();

    if (calculationScalarType.empty()) {
        JST_ERROR("[MODULE_CORRELATOR_NATIVE_CUDA] Unsupported calculation mode '{}'.",
                  calculationMode);
        return Result::ERROR;
    }

    std::string error;

    if (!DeriveCorrelatorLaunchPlan(inputTensor.shape(kAspectAxis),
                                    inputTensor.shape(kFrequencyAxis),
                                    inputTensor.shape(kTimeAxis),
                                    inputTensor.dtype() == DataType::CI8,
                                    calculationMode == "integer",
                                    (calculationMode == "double_precision_fp") ? 8 : 4,
                                    plan,
                                    error)) {
        JST_ERROR("[MODULE_CORRELATOR_NATIVE_CUDA] Cannot correlate shape {}: {}.",
                  inputTensor.shape(), error);
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result CorrelatorImplNativeCuda::computeInitialize() {
    const std::string constants = BuildCorrelatorConstants(plan,
                                                          inputTensor.shape(kAspectAxis),
                                                          inputTensor.shape(kFrequencyAxis),
                                                          inputTensor.shape(kTimeAxis),
                                                          conjugateAntennaIndex);

    const std::unordered_map<std::string, std::string> pieces = {
        {"type_aliases", BuildCorrelatorTypeAliases(inputScalarType, calculationScalarType)},
        {"kernel_constants", constants},
    };

    JST_CHECK(createKernel(kCorrelatorKernelName,
                           plan.packed ? kCorrelatorPackedKernelSource : kCorrelatorGenericKernelSource,
                           pieces));
    kernelCreated = true;

    if (plan.packed) {
        const std::unordered_map<std::string, std::string> accumulatePieces = {
            {"accumulate_constants",
             jst::fmt::format("static constexpr U64 ELEMENT_COUNT = {}ull;", elementCount)},
        };

        JST_CHECK(createKernel(kAccumulateKernelName, kAccumulateKernelSource, accumulatePieces));
        accumulateKernelCreated = true;
    }

    JST_DEBUG("[MODULE_CORRELATOR_NATIVE_CUDA] Compiled {} kernel: grid [{}, {}, {}], block {}, "
              "stage {} samples.",
              plan.packed ? "packed CI8" : "generic",
              inputTensor.shape(kFrequencyAxis), plan.chunkCount, plan.tileBatchCount,
              plan.threadCount, plan.stageSamples);

    return Result::SUCCESS;
}

Result CorrelatorImplNativeCuda::computeSubmit(const cudaStream_t& stream) {
    if (plan.packed && !accumulatorCleared) {
        JST_CUDA_CHECK(cudaMemsetAsync(accumulatorTensor.data(), 0, accumulatorTensor.sizeBytes(), stream), [&] {
            JST_ERROR("[MODULE_CORRELATOR_NATIVE_CUDA] Failed to clear the accumulator buffer: {}.", err);
        });
        accumulatorCleared = true;
    }

    if (integrationStep == 0) {
        JST_CUDA_CHECK(cudaMemsetAsync(outputTensor.data(), 0, outputTensor.sizeBytes(), stream), [&] {
            JST_ERROR("[MODULE_CORRELATOR_NATIVE_CUDA] Failed to clear the correlator output buffer: {}.", err);
        });
    }

    void* outputData = outputTensor.data();
    void* inputArgument = const_cast<void*>(inputTensor.data());

    void* correlatorTarget = plan.packed ? accumulatorTensor.data() : outputData;
    void* arguments[] = {
        &inputArgument,
        &correlatorTarget,
    };

    const Extent3D<U64> block = {plan.threadCount, 1, 1};
    const Extent3D<U64> grid = {inputTensor.shape(kFrequencyAxis), plan.chunkCount, plan.tileBatchCount};

    JST_CHECK(scheduleKernel(kCorrelatorKernelName, stream, grid, block, arguments));

    if (plan.packed) {
        void* accumulateArguments[] = {
            &correlatorTarget,
            &outputData,
        };

        constexpr U64 kAccumulateThreads = 256;
        const Extent3D<U64> accumulateBlock = {kAccumulateThreads, 1, 1};
        const Extent3D<U64> accumulateGrid = {(elementCount + kAccumulateThreads - 1) / kAccumulateThreads, 1, 1};

        JST_CHECK(scheduleKernel(kAccumulateKernelName, stream, accumulateGrid, accumulateBlock,
                                 accumulateArguments));
    }

    integrationStep = (integrationStep + 1) % integrationRate;
    if (integrationStep != 0) {
        return Result::SKIP;
    }

    if (inputTensor.hasAttribute("timestamp")) {
        JST_CHECK(outputTensor.setAttribute("timestamp", inputTensor.attribute("timestamp")));
    }

    return Result::SUCCESS;
}

Result CorrelatorImplNativeCuda::computeDeinitialize() {
    if (kernelCreated) {
        JST_CHECK(destroyKernel(kCorrelatorKernelName));
    }

    if (accumulateKernelCreated) {
        JST_CHECK(destroyKernel(kAccumulateKernelName));
    }

    kernelCreated = false;
    accumulateKernelCreated = false;

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(CorrelatorImplNativeCuda, DeviceType::CUDA, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
