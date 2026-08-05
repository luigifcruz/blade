#include <jetstream/backend/devices/cuda/helpers.hh>
#include <jetstream/memory/macros.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>
#include <jetstream/runtime_context_native_cuda.hh>
#include <jetstream/scheduler_context.hh>

#include <cstdint>
#include <limits>

#include "module_impl.hh"

namespace Jetstream::Modules {

namespace {

constexpr U64 kMaxThreadsPerBlock = 1024;
constexpr U64 kMaxGridSizeX = std::numeric_limits<I32>::max();

constexpr const char* kIntegratorKernelSource = R"(
<<<type_aliases>>>
<<<kernel_constants>>>

extern "C" __global__ void integrator(const InputScalar* input,
                                      float* output,
                                      const U64 output_size) {
    const U64 tid = static_cast<U64>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (tid < output_size) {
        const U64 innerIndex = tid % INTEGRATIONS;
        const U64 integrationGroup = tid / INTEGRATIONS;
        float accumulator[2] = {};

        for (U64 i = 0; i < INTEGRATION_SIZE; i++) {
            const U64 inputElement = (integrationGroup * INTEGRATION_SIZE * INTEGRATIONS) +
                                     (i * INTEGRATIONS) + innerIndex;
            for (U64 component = 0; component < INPUT_COMPONENTS; component++) {
                accumulator[component] +=
                    static_cast<float>(input[(inputElement * INPUT_COMPONENTS) + component]);
            }
        }

        for (U64 component = 0; component < 2; component++) {
            output[(tid * 2) + component] += accumulator[component];
        }
    }
}
)";

constexpr const char* kIntegratorKernelName  = "integrator";

}  // namespace

struct IntegratorImplNativeCuda : public IntegratorImpl,
                                public NativeCudaRuntimeContext,
                                public Scheduler::Context {
 public:
    Result validate() final;
    Result create() final;
    Result computeInitialize() override;
    Result computeSubmit(const cudaStream_t& stream) override;
    Result computeDeinitialize() override;

 private:
    std::string kernelName;
    bool kernelCreated = false;
    U64 validatedComponentCount = 0;
    U64 validatedGridSize = 0;
    U64 componentCount = 1;
    U64 gridSize = 0;
};

Result IntegratorImplNativeCuda::validate() {
    validatedComponentCount = 0;
    validatedGridSize = 0;

    JST_CHECK(IntegratorImpl::validate());

    const auto& config = *candidate();
    if (config.blockSize == 0 || config.blockSize > kMaxThreadsPerBlock) {
        JST_ERROR("[MODULE_INTEGRATOR_NATIVE_CUDA] The CUDA block size must be between 1 and {}.",
                  kMaxThreadsPerBlock);
        return Result::ERROR;
    }

    if (!inputs().contains("buffer")) {
        return Result::SUCCESS;
    }

    const Tensor& input = inputs().at("buffer").tensor;
    if (!input.validShape() || input.size() == 0) {
        return Result::SUCCESS;
    }

    if (
        input.dtype() != DataType::F32 &&
        input.dtype() != DataType::CF32 &&
        input.dtype() != DataType::CI8
    ) {
        JST_ERROR("[MODULE_INTEGRATOR_NATIVE_CUDA] Unsupported input data type '{}'. Expected F32, CF32, or CI8.",
                  input.dtype());
        return Result::ERROR;
    }

    validatedBypass = config.size == 1 && config.rate == 1 &&
                      input.dtype() == DataType::CF32;
    if (validatedBypass) {
        return Result::SUCCESS;
    }

    U64 alignedOutputSize = 0;
    if (!detail::CheckedPageAlignedSize(validatedOutputSizeBytes,
                                        alignedOutputSize) ||
        alignedOutputSize > std::numeric_limits<std::size_t>::max()) {
        JST_ERROR("[MODULE_INTEGRATOR_NATIVE_CUDA] Output allocation size is too large.");
        return Result::ERROR;
    }

    const U64 blockCount =
        validatedNumberOfElements / config.blockSize +
        (validatedNumberOfElements % config.blockSize != 0);
    if (blockCount > kMaxGridSizeX) {
        JST_ERROR("[MODULE_INTEGRATOR_NATIVE_CUDA] Output size exceeds the CUDA grid limit.");
        return Result::ERROR;
    }

    validatedComponentCount = IsDataTypeComplex(input.dtype()) ? 2 : 1;
    validatedGridSize = blockCount;
    return Result::SUCCESS;
}

Result IntegratorImplNativeCuda::create() {
    JST_CHECK(IntegratorImpl::create());

    if (bypass) {
        return Result::SUCCESS;
    }

    componentCount = validatedComponentCount;
    gridSize = validatedGridSize;

    return Result::SUCCESS;
}

Result IntegratorImplNativeCuda::computeInitialize() {
    if (bypass) {
        return Result::SUCCESS;
    }

    const std::string scalarType =
        inputTensor.dtype() == DataType::CI8 ? "signed char" : "float";

    const std::unordered_map<std::string, std::string> pieces = {
        {"type_aliases",
        jst::fmt::format("using U64 = unsigned long long;\n"
                         "using InputScalar = {};",
                         scalarType)},
        {"kernel_constants",
        jst::fmt::format("static constexpr U64 INTEGRATION_SIZE = {};\n"
                         "static constexpr U64 INTEGRATIONS = {};\n"
                         "static constexpr U64 INPUT_COMPONENTS = {};",
                         size,
                         integratedElementCount,
                         componentCount)},
    };
    JST_CHECK(createKernel(kIntegratorKernelName, kIntegratorKernelSource, pieces));
    kernelCreated = true;
    return Result::SUCCESS;
}

Result IntegratorImplNativeCuda::computeSubmit(const cudaStream_t& stream) {
    if (bypass) {
        return Result::SUCCESS;
    }

    if (blockIndex == 0) {
        auto* outputBase = static_cast<std::uint8_t*>(outputTensor.buffer().data());
        if (!outputBase) {
            JST_ERROR("[MODULE_INTEGRATOR_NATIVE_CUDA] Missing output device buffer.");
            return Result::ERROR;
        }
        void* outputData = outputBase + outputTensor.offsetBytes();
        JST_CUDA_CHECK(cudaMemsetAsync(outputData, 0, outputTensor.sizeBytes(), stream), [&] {
            JST_ERROR("[MODULE_INTEGRATOR_NATIVE_CUDA] Failed to clear the output buffer: {}.", err);
        });
    }

    const auto* inputBase = static_cast<const std::uint8_t*>(inputTensor.buffer().data());
    auto* outputBase = static_cast<std::uint8_t*>(outputTensor.buffer().data());
    if (!inputBase || !outputBase) {
        JST_ERROR("[MODULE_INTEGRATOR_NATIVE_CUDA] Missing input or output device buffer.");
        return Result::ERROR;
    }
    const void* inputData = inputBase + inputTensor.offsetBytes();
    void* outputData = outputBase + outputTensor.offsetBytes();

    void* inputArgument = const_cast<void*>(inputData);
    void* arguments[] = {
        &inputArgument,
        &outputData,
        (void*)&numberOfElements
    };

    const Extent3D<U64> block = {blockSize, 1, 1};
    const Extent3D<U64> grid = {gridSize, 1, 1};

    JST_CHECK(scheduleKernel(kIntegratorKernelName, stream, grid, block, arguments));
    blockIndex = (blockIndex + 1) % rate;

    return blockIndex == 0 ? Result::SUCCESS : Result::SKIP;
}

Result IntegratorImplNativeCuda::computeDeinitialize() {
    if (kernelCreated) {
        JST_CHECK(destroyKernel(kIntegratorKernelName));
    }

    kernelCreated = false;

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(IntegratorImplNativeCuda, DeviceType::CUDA, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
