#include <cstddef>
#include <cstdint>
#include <limits>
#include <unordered_map>

#include <jetstream/backend/devices/cuda/helpers.hh>
#include <jetstream/memory/macros.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>
#include <jetstream/runtime_context_native_cuda.hh>
#include <jetstream/scheduler_context.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

namespace {

constexpr U64 kMaxGridSizeX = std::numeric_limits<I32>::max();
constexpr U64 kMaxGridSizeY = 65535;
constexpr U64 kSharedMemoryBudget = 49152;
constexpr const char* kBeamformerKernelName = "beamformer_ata";
constexpr const char* kBeamformerKernelSource = R"(
<<<type_aliases>>>
<<<kernel_constants>>>

struct alignas(2 * sizeof(InputScalar)) InputComplex {
    InputScalar real;
    InputScalar imag;
};

struct alignas(8) Complex {
    float real;
    float imag;

    __device__ Complex() : real(0.0f), imag(0.0f) {}

    __device__ Complex(float realValue, float imagValue)
        : real(realValue), imag(imagValue) {}
};

struct alignas(16) ComplexPair {
    Complex x;
    Complex y;
};

static_assert(NPOLS == 2, "Beamformer expects exactly two polarizations.");
static_assert(sizeof(InputComplex) == 2 * sizeof(InputScalar), "Beamformer input complex layout is invalid.");
static_assert(sizeof(Complex) == 8, "Beamformer complex layout must remain 8 bytes.");
static_assert(alignof(Complex) == 8, "Beamformer complex alignment must remain 8 bytes.");
static_assert(sizeof(ComplexPair) == 16, "Beamformer output pair layout must remain 16 bytes.");
static_assert(alignof(ComplexPair) == 16, "Beamformer output pair alignment must remain 16 bytes.");

__device__ Complex detect(const Complex& value) {
    return Complex((value.real * value.real) + (value.imag * value.imag), 0.0f);
}

__device__ Complex convert(const InputComplex& value) {
    return Complex(static_cast<float>(value.real) * INPUT_SCALE,
                   static_cast<float>(value.imag) * INPUT_SCALE);
}

__device__ Complex multiply(const Complex& lhs, const Complex& rhs) {
    return Complex((lhs.real * rhs.real) - (lhs.imag * rhs.imag),
                   (lhs.real * rhs.imag) + (lhs.imag * rhs.real));
}

__device__ Complex add(const Complex& lhs, const Complex& rhs) {
    return Complex(lhs.real + rhs.real, lhs.imag + rhs.imag);
}

extern "C" __global__ void beamformer_ata(const InputComplex* input,
                                           const Complex* phasor,
                                           Complex* out) {
    const unsigned long long bi = threadIdx.x;
    const unsigned long long ti = bi + (blockIdx.y * TBLOCK);
    const unsigned long long ch = blockIdx.x;

    __shared__ Complex phr_cache[NBEAMS][NANTS][NPOLS];

    unsigned long long iy = (ch * NPOLS) + (bi * NPOLS * NCHANS * NANTS);
    const unsigned long long dy = NPOLS * NCHANS;

    if (bi < NBEAMS) {
        for (unsigned long long a = 0; a < NANTS; a++, iy += dy) {
            phr_cache[bi][a][0] = phasor[iy + 0];
            phr_cache[bi][a][1] = phasor[iy + 1];
        }
    }

    __syncthreads();

    Complex ant_cache[NANTS][NPOLS];

    unsigned long long ix = (ch * NTIME * NPOLS) + (ti * NPOLS);
    const unsigned long long dx = NTIME * NCHANS * NPOLS;

    for (unsigned long long a = 0; a < NANTS; a++, ix += dx) {
        ant_cache[a][0] = convert(input[ix + 0]);
        ant_cache[a][1] = convert(input[ix + 1]);
    }

    unsigned long long iz = (ch * NTIME) + ti;
    const unsigned long long dz = NTIME * NCHANS;

    for (unsigned long long b = 0; b < NBEAMS; b++, iz += dz) {
        Complex acc[NPOLS] = {Complex(0.0f, 0.0f), Complex(0.0f, 0.0f)};

        for (unsigned long long a = 0; a < NANTS; a++) {
            acc[0] = add(acc[0], multiply(ant_cache[a][0], phr_cache[b][a][0]));
            acc[1] = add(acc[1], multiply(ant_cache[a][1], phr_cache[b][a][1]));
        }

        reinterpret_cast<ComplexPair*>(out)[iz] = ComplexPair{acc[0], acc[1]};
    }

    if (ENABLE_INCOHERENT_BEAM) {
        Complex acc[NPOLS] = {Complex(0.0f, 0.0f), Complex(0.0f, 0.0f)};

        for (unsigned long long a = 0; a < NANTS; a++) {
            acc[0] = add(acc[0], detect(multiply(ant_cache[a][0], phr_cache[0][a][0])));
            acc[1] = add(acc[1], detect(multiply(ant_cache[a][1], phr_cache[0][a][1])));
        }

        if (ENABLE_INCOHERENT_BEAM_SQRT) {
            acc[0] = Complex(sqrtf(acc[0].real), acc[0].imag);
            acc[1] = Complex(sqrtf(acc[1].real), acc[1].imag);
        }

        reinterpret_cast<ComplexPair*>(out)[iz] = ComplexPair{acc[0], acc[1]};
    }
}
)";

}  // namespace

struct BeamformerImplNativeCuda : public BeamformerImpl,
                                  public NativeCudaRuntimeContext,
                                  public Scheduler::Context {
 public:
    Result validate() final;
    Result create() final;
    Result computeInitialize() override;
    Result computeSubmit(const cudaStream_t& stream) override;
    Result computeDeinitialize() override;

 private:
    bool kernelCreated = false;
};

Result BeamformerImplNativeCuda::validate() {
    JST_CHECK(BeamformerImpl::validate());

    const auto& config = *candidate();
    if (config.blockSize == 0 || config.blockSize > 1024) {
        JST_ERROR("[MODULE_BEAMFORMER_NATIVE_CUDA] The CUDA block size must be between 1 and 1024.");
        return Result::ERROR;
    }

    if (!inputs().contains("buffer") || !inputs().contains("phasors")) {
        return Result::SUCCESS;
    }

    const Tensor& input = inputs().at("buffer").tensor;
    const Tensor& phasors = inputs().at("phasors").tensor;
    if (!input.validShape() || input.size() == 0 ||
        !phasors.validShape() || phasors.size() == 0) {
        return Result::SUCCESS;
    }

    if (input.dtype() != DataType::CI8 && input.dtype() != DataType::CF32) {
        JST_ERROR("[MODULE_BEAMFORMER_NATIVE_CUDA] Unsupported input data type '{}'. Expected CI8 or CF32.",
                  input.dtype());
        return Result::ERROR;
    }

    if (phasors.dtype() != DataType::CF32) {
        JST_ERROR("[MODULE_BEAMFORMER_NATIVE_CUDA] Unsupported phasor data type '{}'. Expected CF32.",
                  phasors.dtype());
        return Result::ERROR;
    }

    if ((input.shape(kBufferTimeAxis) % config.blockSize) != 0) {
        JST_ERROR("[MODULE_BEAMFORMER_NATIVE_CUDA] Number of time samples ({}) isn't divisible by the block size ({}).",
                  input.shape(kBufferTimeAxis),
                  config.blockSize);
        return Result::ERROR;
    }

    if (phasors.shape(kPhasorBeamAxis) > config.blockSize) {
        JST_ERROR("[MODULE_BEAMFORMER_NATIVE_CUDA] The block size ({}) is smaller than the number of beams ({}).",
                  config.blockSize,
                  phasors.shape(kPhasorBeamAxis));
        return Result::ERROR;
    }

    if (input.shape(kBufferFrequencyAxis) > kMaxGridSizeX ||
        input.shape(kBufferTimeAxis) / config.blockSize > kMaxGridSizeY) {
        JST_ERROR("[MODULE_BEAMFORMER_NATIVE_CUDA] Input shape exceeds the CUDA grid limits.");
        return Result::ERROR;
    }

    const U64 maxKernelIndex = std::numeric_limits<I32>::max();
    U64 outputPairCount = 1;
    for (Index axis = 0; axis < kBufferPolarizationAxis; ++axis) {
        if (!detail::CheckedMultiply(outputPairCount,
                                     validatedOutputShape[axis],
                                     outputPairCount)) {
            JST_ERROR("[MODULE_BEAMFORMER_NATIVE_CUDA] Output indexing exceeds CUDA limits.");
            return Result::ERROR;
        }
    }
    if (input.size() > maxKernelIndex || phasors.size() > maxKernelIndex ||
        outputPairCount > maxKernelIndex) {
        JST_ERROR("[MODULE_BEAMFORMER_NATIVE_CUDA] Tensor indexing exceeds the kernel's 32-bit range.");
        return Result::ERROR;
    }

    U64 sharedPhasorCount = 0;
    U64 sharedPhasorSizeBytes = 0;
    if (!detail::CheckedMultiply(phasors.shape(kPhasorBeamAxis),
                                 input.shape(kBufferAspectAxis),
                                 sharedPhasorCount) ||
        !detail::CheckedMultiply(sharedPhasorCount,
                                 kExpectedPolarizations,
                                 sharedPhasorCount) ||
        !detail::CheckedMultiply(sharedPhasorCount,
                                 static_cast<U64>(sizeof(CF32)),
                                 sharedPhasorSizeBytes) ||
        sharedPhasorSizeBytes > kSharedMemoryBudget) {
        JST_ERROR("[MODULE_BEAMFORMER_NATIVE_CUDA] Phasor cache exceeds the CUDA shared-memory limit.");
        return Result::ERROR;
    }

    U64 alignedOutputSize = 0;
    if (!detail::CheckedPageAlignedSize(validatedOutputSizeBytes,
                                        alignedOutputSize) ||
        alignedOutputSize > std::numeric_limits<std::size_t>::max()) {
        JST_ERROR("[MODULE_BEAMFORMER_NATIVE_CUDA] Output allocation size is too large.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result BeamformerImplNativeCuda::create() {
    JST_CHECK(BeamformerImpl::create());

    return Result::SUCCESS;
}

Result BeamformerImplNativeCuda::computeInitialize() {
    const std::string inputScalarType = [&]() -> std::string {
        switch (inputTensor.dtype()) {
            case DataType::CI8: return "signed char";
            case DataType::CF32: return "float";
            default: return "";
        }
    }();

    if (inputScalarType.empty()) {
        JST_ERROR("[MODULE_BEAMFORMER_NATIVE_CUDA] Unsupported input data type '{}'. Expected CI8 or CF32.",
                  inputTensor.dtype());
        return Result::ERROR;
    }

    const std::string inputScale = inputTensor.dtype() == DataType::CI8
        ? "(1.0f / 128.0f)"
        : "1.0f";

    const std::unordered_map<std::string, std::string> pieces = {
        {"type_aliases",
         jst::fmt::format("using InputScalar = {};\n"
                          "static constexpr float INPUT_SCALE = {};",
                          inputScalarType,
                          inputScale)},
        {"kernel_constants",
         jst::fmt::format("static constexpr int NBEAMS = {};\n"
                          "static constexpr int NANTS = {};\n"
                          "static constexpr int NCHANS = {};\n"
                          "static constexpr int NTIME = {};\n"
                          "static constexpr int NPOLS = {};\n"
                          "static constexpr int TBLOCK = {};\n"
                          "static constexpr bool ENABLE_INCOHERENT_BEAM = {};\n"
                          "static constexpr bool ENABLE_INCOHERENT_BEAM_SQRT = {};",
                          phasorTensor.shape(kPhasorBeamAxis),
                          inputTensor.shape(kBufferAspectAxis),
                          inputTensor.shape(kBufferFrequencyAxis),
                          inputTensor.shape(kBufferTimeAxis),
                          inputTensor.shape(kBufferPolarizationAxis),
                          blockSize,
                          enableIncoherentBeam ? "true" : "false",
                          enableIncoherentBeamSqrt ? "true" : "false")},
    };

    JST_CHECK(createKernel(kBeamformerKernelName, kBeamformerKernelSource, pieces));
    kernelCreated = true;

    return Result::SUCCESS;
}

Result BeamformerImplNativeCuda::computeSubmit(const cudaStream_t& stream) {
    const auto* inputBase = static_cast<const std::uint8_t*>(inputTensor.buffer().data());
    const auto* phasorBase = static_cast<const std::uint8_t*>(phasorTensor.buffer().data());
    auto* outputBase = static_cast<std::uint8_t*>(outputTensor.buffer().data());
    const void* inputData = inputBase + inputTensor.offsetBytes();
    const void* phasorData = phasorBase + phasorTensor.offsetBytes();
    void* outputData = outputBase + outputTensor.offsetBytes();

    JST_CUDA_CHECK(cudaMemsetAsync(outputData, 0, outputTensor.sizeBytes(), stream), [&] {
        JST_ERROR("[MODULE_BEAMFORMER_NATIVE_CUDA] Failed to clear the beamformer output buffer: {}.", err);
    });

    void* inputArgument = const_cast<void*>(inputData);
    void* phasorArgument = const_cast<void*>(phasorData);
    void* arguments[] = {
        &inputArgument,
        &phasorArgument,
        &outputData,
    };

    const Extent3D<U64> block = {blockSize, 1, 1};
    const Extent3D<U64> grid = {
        inputTensor.shape(kBufferFrequencyAxis),
        inputTensor.shape(kBufferTimeAxis) / blockSize,
        1,
    };

    JST_CHECK(scheduleKernel(kBeamformerKernelName, stream, grid, block, arguments));

    return Result::SUCCESS;
}

Result BeamformerImplNativeCuda::computeDeinitialize() {
    if (kernelCreated) {
        JST_CHECK(destroyKernel(kBeamformerKernelName));
    }

    kernelCreated = false;

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(BeamformerImplNativeCuda, DeviceType::CUDA, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
