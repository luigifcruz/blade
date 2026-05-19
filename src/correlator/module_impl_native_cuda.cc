#include <jetstream/backend/devices/cuda/helpers.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>
#include <jetstream/runtime_context_native_cuda.hh>
#include <jetstream/scheduler_context.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

namespace {

constexpr const char* kCorrelatorKernelName = "correlator";
constexpr const char* kCorrelatorKernelSource = R"(
<<<type_aliases>>>
<<<kernel_constants>>>

struct alignas(2 * sizeof(InputScalar)) InputComplex {
    InputScalar real;
    InputScalar imag;
};

struct alignas(2 * sizeof(CalcScalar)) CalcComplex {
    CalcScalar real;
    CalcScalar imag;

    __device__ CalcComplex() : real(0), imag(0) {}

    __device__ CalcComplex(CalcScalar realValue, CalcScalar imagValue)
        : real(realValue), imag(imagValue) {}

    __device__ explicit CalcComplex(const InputComplex& rhs)
        : real(static_cast<CalcScalar>(rhs.real)),
          imag(static_cast<CalcScalar>(rhs.imag)) {}
};

struct alignas(2 * sizeof(OutputScalar)) OutputComplex {
    OutputScalar real;
    OutputScalar imag;

    __device__ OutputComplex() : real(0.0f), imag(0.0f) {}

    __device__ OutputComplex(OutputScalar realValue, OutputScalar imagValue)
        : real(realValue), imag(imagValue) {}

    __device__ explicit OutputComplex(const CalcComplex& rhs)
        : real(static_cast<OutputScalar>(rhs.real)),
          imag(static_cast<OutputScalar>(rhs.imag)) {}

    __device__ __forceinline__ OutputComplex& operator+=(const OutputComplex& rhs) {
        real += rhs.real;
        imag += rhs.imag;
        return *this;
    }
};

__device__ __forceinline__ CalcComplex multiplyConjugate(const CalcComplex& lhs, const CalcComplex& rhs) {
    return CalcComplex((lhs.real * rhs.real) + (lhs.imag * rhs.imag),
                       (lhs.imag * rhs.real) - (lhs.real * rhs.imag));
}

__device__ __forceinline__ void atomicAddComplex(OutputComplex* value, const OutputComplex& delta) {
    atomicAdd(&value->real, delta.real);
    atomicAdd(&value->imag, delta.imag);
}

extern "C" __global__ void correlator(const InputComplex* input, OutputComplex* output) {
    const U64 BIX = blockIdx.x;
    const U64 BIY = blockIdx.y;

    const U64 TIX = threadIdx.x;
    const U64 TIY = threadIdx.y;

    const U64 OUTPUT_POLS = 4;
    const U64 AAI = BIX;
    const U64 CI = TIX + (BIY * BLOCK_SIZE_X);
    constexpr U64 TIME_CHUNK_SIZE = T / BLOCK_SIZE_Y;
    const U64 TIME_INDEX_OFFSET = TIY * TIME_CHUNK_SIZE;

    InputComplex (*reference)[P] = nullptr;
    <<<shared_memory_setup>>>

    if constexpr (USE_SHARED_MEMORY) {
        for (U64 TI = 0; TI < TIME_CHUNK_SIZE; TI++) {
            const U64 ANTENNA_A_INDEX = (AAI * C * T * P) + (CI * T * P) + ((TI + TIME_INDEX_OFFSET) * P);
            reference[TI + TIME_INDEX_OFFSET][0] = input[ANTENNA_A_INDEX + 0];
            reference[TI + TIME_INDEX_OFFSET][1] = input[ANTENNA_A_INDEX + 1];
        }
        __syncthreads();
    }

    for (U64 ABI = AAI; ABI < A; ABI++) {
        const U64 BASELINE_INDEX = ((AAI * (2 * A - AAI + 1)) / 2) + (ABI - AAI);

        OutputComplex sumXX = OutputComplex(0.0f, 0.0f);
        OutputComplex sumXY = OutputComplex(0.0f, 0.0f);
        OutputComplex sumYX = OutputComplex(0.0f, 0.0f);
        OutputComplex sumYY = OutputComplex(0.0f, 0.0f);

        for (U64 TI = 0; TI < TIME_CHUNK_SIZE; TI++) {
            CalcComplex AVAX;
            CalcComplex AVAY = CalcComplex();
            CalcComplex AVBX;
            CalcComplex AVBY = CalcComplex();

            if constexpr (USE_SHARED_MEMORY) {
                AVAX = CalcComplex(reference[TI + TIME_INDEX_OFFSET][0]);
                AVAY = CalcComplex(reference[TI + TIME_INDEX_OFFSET][1]);
            } else {
                const U64 ANTENNA_A_INDEX = (AAI * C * T * P) + (CI * T * P) + ((TI + TIME_INDEX_OFFSET) * P);
                AVAX = CalcComplex(input[ANTENNA_A_INDEX + 0]);
                AVAY = CalcComplex(input[ANTENNA_A_INDEX + 1]);
            }

            const U64 ANTENNA_B_INDEX = (ABI * C * T * P) + (CI * T * P) + ((TI + TIME_INDEX_OFFSET) * P);
            AVBX = CalcComplex(input[ANTENNA_B_INDEX + 0]);
            AVBY = CalcComplex(input[ANTENNA_B_INDEX + 1]);

            if constexpr (CONJUGATE_ANTENNA == 1) {
                sumXX += OutputComplex(multiplyConjugate(AVAX, AVBX));
                sumXY += OutputComplex(multiplyConjugate(AVAX, AVBY));
                sumYX += OutputComplex(multiplyConjugate(AVAY, AVBX));
                sumYY += OutputComplex(multiplyConjugate(AVAY, AVBY));
            } else {
                sumXX += OutputComplex(multiplyConjugate(AVBX, AVAX));
                sumXY += OutputComplex(multiplyConjugate(AVBY, AVAX));
                sumYX += OutputComplex(multiplyConjugate(AVBX, AVAY));
                sumYY += OutputComplex(multiplyConjugate(AVBY, AVAY));
            }
        }

        const U64 OUTPUT_INDEX = (BASELINE_INDEX * C * OUTPUT_POLS) + (CI * OUTPUT_POLS);

        atomicAddComplex(output + OUTPUT_INDEX + 0, sumXX);
        atomicAddComplex(output + OUTPUT_INDEX + 1, sumXY);
        atomicAddComplex(output + OUTPUT_INDEX + 2, sumYX);
        atomicAddComplex(output + OUTPUT_INDEX + 3, sumYY);
    }
}
)";

}  // namespace

struct CorrelatorImplNativeCuda : public CorrelatorImpl,
                                  public NativeCudaRuntimeContext,
                                  public Scheduler::Context {
 public:
    Result create() final;
    Result computeInitialize() override;
    Result computeSubmit(const cudaStream_t& stream) override;
    Result computeDeinitialize() override;

 private:
    bool kernelCreated = false;
};

Result CorrelatorImplNativeCuda::create() {
    const Tensor& input = inputs().at("buffer").tensor;

    if (input.dtype() != DataType::CI8 && input.dtype() != DataType::CF32) {
        JST_ERROR("[MODULE_CORRELATOR_NATIVE_CUDA] Unsupported input data type '{}'. Expected CI8 or CF32.",
                  input.dtype());
        return Result::ERROR;
    }

    JST_CHECK(CorrelatorImpl::create());

    return Result::SUCCESS;
}

Result CorrelatorImplNativeCuda::computeInitialize() {
    const std::string inputScalarType = [&]() -> std::string {
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

    const std::string calculationScalarType = [&]() -> std::string {
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

    const std::unordered_map<std::string, std::string> pieces = {
        {"type_aliases",
         jst::fmt::format("using U64 = unsigned long long;\n"
                          "using InputScalar = {};\n"
                          "using CalcScalar = {};\n"
                          "using OutputScalar = float;",
                          inputScalarType,
                          calculationScalarType)},
        {"kernel_constants",
         jst::fmt::format("static constexpr U64 A = {}ull;\n"
                          "static constexpr U64 C = {}ull;\n"
                          "static constexpr U64 T = {}ull;\n"
                          "static constexpr U64 P = {}ull;\n"
                          "static constexpr U64 BLOCK_SIZE_X = {}ull;\n"
                          "static constexpr U64 BLOCK_SIZE_Y = {}ull;\n"
                          "static constexpr U64 CONJUGATE_ANTENNA = {}ull;\n"
                          "static constexpr bool USE_SHARED_MEMORY = {};",
                          inputTensor.shape(kAspectAxis),
                          inputTensor.shape(kFrequencyAxis),
                          inputTensor.shape(kTimeAxis),
                          inputTensor.shape(kPolarizationAxis),
                          blockSizeX,
                          blockSizeY,
                          conjugateAntennaIndex,
                          sharedMemoryEnabled ? "true" : "false")},
        {"shared_memory_setup",
         sharedMemoryEnabled ? "__shared__ InputComplex sharedReference[T][P];\nreference = sharedReference;"
                             : ""},
    };

    JST_CHECK(createKernel(kCorrelatorKernelName, kCorrelatorKernelSource, pieces));
    kernelCreated = true;

    return Result::SUCCESS;
}

Result CorrelatorImplNativeCuda::computeSubmit(const cudaStream_t& stream) {
    if (integrationStep == 0) {
        JST_CUDA_CHECK(cudaMemsetAsync(outputTensor.data(), 0, outputTensor.sizeBytes(), stream), [&] {
            JST_ERROR("[MODULE_CORRELATOR_NATIVE_CUDA] Failed to clear the correlator output buffer: {}.", err);
        });
    }

    const void* inputData = inputTensor.data();
    void* outputData = outputTensor.data();

    void* inputArgument = const_cast<void*>(inputData);
    void* arguments[] = {
        &inputArgument,
        &outputData,
    };

    const Extent3D<U64> block = {blockSizeX, blockSizeY, 1};
    const Extent3D<U64> grid = {inputTensor.shape(kAspectAxis), inputTensor.shape(kFrequencyAxis) / blockSizeX, 1};

    JST_CHECK(scheduleKernel(kCorrelatorKernelName, stream, grid, block, arguments));

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

    kernelCreated = false;

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(CorrelatorImplNativeCuda, DeviceType::CUDA, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
