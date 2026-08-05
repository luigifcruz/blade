#include <jetstream/backend/devices/cuda/helpers.hh>
#include <jetstream/memory/macros.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>
#include <jetstream/runtime_context_native_cuda.hh>
#include <jetstream/scheduler_context.hh>

#include "module_impl.hh"

#include <cstdint>
#include <limits>

namespace Jetstream::Modules {

namespace {

constexpr const char* kStackerKernelSource = R"(
using U64 = unsigned long long;

<<<kernel_constants>>>

struct alignas(ELEMENT_SIZE) Element {
    unsigned char data[ELEMENT_SIZE];
};

extern "C" __global__ void stacker(const Element* input,
                                   Element* output,
                                   const U64 input_size,
                                   const U64 stackIndex) {
    const U64 tid = static_cast<U64>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (tid < input_size) {
        const U64 oid = (tid / WIDTH_IN) * WIDTH_OUT +
                        (stackIndex * WIDTH_IN) + (tid % WIDTH_IN);
        output[oid] = input[tid];
    }
}
)";

constexpr const char* kStackerKernelName  = "stacker";
constexpr U64 kMaxGridSizeX = std::numeric_limits<I32>::max();
constexpr U64 kMaxCudaPitch = std::numeric_limits<I32>::max();

}  // namespace

struct StackerImplNativeCuda : public StackerImpl,
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
    bool validatedKernelNotCopy = false;
    bool kernelNotCopy = false;
    U64 validatedBlockSize = 0;
    U64 launchBlockSize = 0;
};

Result StackerImplNativeCuda::validate() {
    validatedKernelNotCopy = false;
    validatedBlockSize = 0;
    JST_CHECK(StackerImpl::validate());

    if (!inputs().contains("buffer")) {
        return Result::SUCCESS;
    }

    const Tensor& input = inputs().at("buffer").tensor;
    if (!input.validShape() || input.size() == 0) {
        return Result::SUCCESS;
    }
    if (input.dtype() != DataType::F32 &&
        input.dtype() != DataType::CF32 &&
        input.dtype() != DataType::CI8) {
        JST_ERROR("[MODULE_STACKER_NATIVE_CUDA] Unsupported input data type '{}'. Expected F32, CF32, or CI8.",
                  input.dtype());
        return Result::ERROR;
    }

    if (validatedBypass) {
        return Result::SUCCESS;
    }

    U64 alignedOutputSize = 0;
    if (!detail::CheckedPageAlignedSize(validatedOutputSizeBytes,
                                         alignedOutputSize) ||
        alignedOutputSize > std::numeric_limits<std::size_t>::max()) {
        JST_ERROR("[MODULE_STACKER_NATIVE_CUDA] Output allocation size is too large.");
        return Result::ERROR;
    }

    const auto& config = *candidate();
    const bool candidateKernelNotCopy =
        validatedWidth < config.copySizeThreshold;
    if (candidateKernelNotCopy) {
        if (config.blockSize == 0 || config.blockSize > 1024) {
            JST_ERROR("[MODULE_STACKER_NATIVE_CUDA] The CUDA block size must be between 1 and 1024 for the kernel path.");
            return Result::ERROR;
        }

        const U64 blockCount = validatedInputSize / config.blockSize +
                               (validatedInputSize % config.blockSize != 0);
        if (blockCount > kMaxGridSizeX) {
            JST_ERROR("[MODULE_STACKER_NATIVE_CUDA] Input size exceeds the CUDA grid limit.");
            return Result::ERROR;
        }
        validatedBlockSize = config.blockSize;
    } else if (validatedWidthByteSize > kMaxCudaPitch ||
               validatedOutputRowByteSize > kMaxCudaPitch ||
               validatedWidthByteSize > std::numeric_limits<std::size_t>::max() ||
               validatedOutputRowByteSize > std::numeric_limits<std::size_t>::max() ||
               validatedHeight > std::numeric_limits<std::size_t>::max()) {
        JST_ERROR("[MODULE_STACKER_NATIVE_CUDA] CUDA copy dimensions exceed the supported range.");
        return Result::ERROR;
    }

    validatedKernelNotCopy = candidateKernelNotCopy;

    return Result::SUCCESS;
}

Result StackerImplNativeCuda::create() {
    JST_CHECK(StackerImpl::create());

    kernelNotCopy = validatedKernelNotCopy;
    launchBlockSize = validatedBlockSize;
    JST_DEBUG("[MODULE_STACKER_NATIVE_CUDA] Height of {} and width of {} elements ({} bytes).",
              height, width, widthByteSize)
    JST_DEBUG("[MODULE_STACKER_NATIVE_CUDA] Stacking with {}.", kernelNotCopy ? "kernel" : "CUDA memcopy");

    return Result::SUCCESS;
}

Result StackerImplNativeCuda::computeInitialize() {
    if (bypass) {
        return Result::SUCCESS;
    }

    if (kernelNotCopy) {
        const std::unordered_map<std::string, std::string> pieces = {
            {"kernel_constants",
            jst::fmt::format("static constexpr U64 ELEMENT_SIZE = {};\n"
                            "static constexpr U64 WIDTH_IN = {};\n"
                            "static constexpr U64 WIDTH_OUT = {};",
                             inputTensor.elementSize(),
                             width,
                             outputWidth)},
        };
        JST_CHECK(createKernel(kStackerKernelName, kStackerKernelSource, pieces));
        kernelCreated = true;
    }
    return Result::SUCCESS;
}

Result StackerImplNativeCuda::computeSubmit(const cudaStream_t& stream) {
    if (bypass) {
        return Result::SUCCESS;
    }

    const auto* inputBase = static_cast<const std::uint8_t*>(inputTensor.buffer().data());
    auto* outputBase = static_cast<std::uint8_t*>(outputTensor.buffer().data());
    if (!inputBase || !outputBase) {
        JST_ERROR("[MODULE_STACKER_NATIVE_CUDA] Missing input or output device buffer.");
        return Result::ERROR;
    }

    const void* inputData = inputBase + inputTensor.offsetBytes();
    void* outputData = outputBase + outputTensor.offsetBytes();

    if (stackIndex == 0) {
        JST_CUDA_CHECK(cudaMemsetAsync(outputData, 0, outputTensor.sizeBytes(), stream), [&] {
            JST_ERROR("[MODULE_STACKER_NATIVE_CUDA] Failed to clear the output buffer: {}.", err);
        });
    }

    if (kernelNotCopy) {
        void* inputArgument = const_cast<void*>(inputData);
        void* arguments[] = {
            &inputArgument,
            &outputData,
            (void*)&inputSize,
            (void*)&stackIndex
        };

        const Extent3D<U64> block = {launchBlockSize, 1, 1};
        const Extent3D<U64> grid = {
            inputSize / launchBlockSize + (inputSize % launchBlockSize != 0),
            1,
            1,
        };

        JST_CHECK(scheduleKernel(kStackerKernelName, stream, grid, block, arguments));
    } else {
        auto* outputBytes = static_cast<std::uint8_t*>(outputData);
        const auto* inputBytes = static_cast<const std::uint8_t*>(inputData);
        JST_CUDA_CHECK(
            cudaMemcpy2DAsync(
                outputBytes + (widthByteSize * stackIndex),
                outputRowByteSize,
                inputBytes,
                widthByteSize,
                widthByteSize,
                height,
                cudaMemcpyDeviceToDevice,
                stream
            ), [&] {
            JST_ERROR("[MODULE_STACKER_NATIVE_CUDA] Failed to copy to the output buffer: {}.", err);
        });
    }

    stackIndex = (stackIndex + 1) % stackRatio;

    return stackIndex == 0 ? Result::SUCCESS : Result::SKIP;
}

Result StackerImplNativeCuda::computeDeinitialize() {
    if (kernelCreated) {
        JST_CHECK(destroyKernel(kStackerKernelName));
    }

    kernelCreated = false;

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(StackerImplNativeCuda, DeviceType::CUDA, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
