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

constexpr const char* kPolarizerXYtoLRKernelSource = R"(
<<<type_aliases>>>

struct alignas(2 * sizeof(Scalar)) Complex {
    Scalar real;
    Scalar imag;

    __device__ Complex(Scalar realValue, Scalar imagValue)
        : real(realValue), imag(imagValue) {}
};


__device__ Complex add(const Complex& lhs, const Complex& rhs) {
    return Complex(lhs.real + rhs.real, lhs.imag + rhs.imag);
}

__device__ Complex sub(const Complex& lhs, const Complex& rhs) {
    return Complex(lhs.real - rhs.real, lhs.imag - rhs.imag);
}

extern "C" __global__ void polarizer_xy_lr(const Complex* input,
                                           Complex* output,
                                           U64 workItemCount) {
    const U64 index = static_cast<U64>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index < workItemCount) {
        const U64 offset = index * 2;
        // The complex multiplication below can be simplified because
        // the real part of the phasor is 0.0. Boring implementation:
        // const IT yPol90 = cuCmulf(yPol, make_cuFloatComplex(0.0, 1.0));

        const Complex xPol = input[offset + 0];
        const Complex yPol = input[offset + 1];

        const Complex yPol90(-yPol.imag, +yPol.real);

        output[offset + 0] = add(xPol, yPol90);
        output[offset + 1] = sub(xPol, yPol90);
    }
}
)";

constexpr const char* kPolarizerXYtoXKernelSource = R"(
<<<type_aliases>>>

struct alignas(2 * sizeof(Scalar)) Complex {
    Scalar real;
    Scalar imag;
};

extern "C" __global__ void polarizer_xy_x(const Complex* input,
                                          Complex* output,
                                          U64 workItemCount) {
    const U64 tid = static_cast<U64>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (tid < workItemCount) {
        output[tid] = input[(tid * 2) + 0];
    }
}
)";

constexpr const char* kPolarizerXYtoYKernelSource = R"(
<<<type_aliases>>>

struct alignas(2 * sizeof(Scalar)) Complex {
    Scalar real;
    Scalar imag;
};

extern "C" __global__ void polarizer_xy_y(const Complex* input,
                                          Complex* output,
                                          U64 workItemCount) {
    const U64 tid = static_cast<U64>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (tid < workItemCount) {
        output[tid] = input[(tid * 2) + 1];
    }
}
)";

constexpr const char* kPolarizerXYtoLRKernelName  = "polarizer_xy_lr";
constexpr const char* kPolarizerXYtoXKernelName = "polarizer_xy_x";
constexpr const char* kPolarizerXYtoYKernelName = "polarizer_xy_y";

}  // namespace

struct PolarizerImplNativeCuda : public PolarizerImpl,
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
    U64 validatedBlockSize = 0;
    U64 launchBlockSize = 0;
};

Result PolarizerImplNativeCuda::validate() {
    validatedBlockSize = 0;
    JST_CHECK(PolarizerImpl::validate());

    const U64 candidateBlockSize = candidate()->blockSize;
    if (validatedPath != PolarizerPath::BYPASS &&
        (candidateBlockSize == 0 || candidateBlockSize > 1024)) {
        JST_ERROR("[MODULE_POLARIZER_NATIVE_CUDA] The CUDA block size must be between 1 and 1024.");
        return Result::ERROR;
    }

    if (!inputs().contains("buffer")) {
        return Result::SUCCESS;
    }

    const Tensor& input = inputs().at("buffer").tensor;
    if (!input.validShape() || input.size() == 0) {
        return Result::SUCCESS;
    }
    if (input.dtype() != DataType::CF32) {
        JST_ERROR("[MODULE_POLARIZER_NATIVE_CUDA] Unsupported input data type '{}'. Expected CF32.", input.dtype());
        return Result::ERROR;
    }

    if (validatedPath == PolarizerPath::BYPASS) {
        return Result::SUCCESS;
    }

    U64 alignedOutputSize = 0;
    if (!detail::CheckedPageAlignedSize(validatedOutputSizeBytes,
                                         alignedOutputSize) ||
        alignedOutputSize > std::numeric_limits<std::size_t>::max()) {
        JST_ERROR("[MODULE_POLARIZER_NATIVE_CUDA] Output allocation size is too large.");
        return Result::ERROR;
    }

    constexpr U64 kMaxGridSizeX = std::numeric_limits<I32>::max();
    const U64 blockCount = validatedOutputWorkItemCount / candidateBlockSize +
                           (validatedOutputWorkItemCount % candidateBlockSize != 0);
    if (blockCount > kMaxGridSizeX) {
        JST_ERROR("[MODULE_POLARIZER_NATIVE_CUDA] Output size exceeds the CUDA grid limit.");
        return Result::ERROR;
    }

    validatedBlockSize = candidateBlockSize;

    return Result::SUCCESS;
}

Result PolarizerImplNativeCuda::create() {
    JST_CHECK(PolarizerImpl::create());
    launchBlockSize = validatedBlockSize;

    return Result::SUCCESS;
}

Result PolarizerImplNativeCuda::computeInitialize() {
    if (bypass) {
        return Result::SUCCESS;
    }

    const std::unordered_map<std::string, std::string> pieces = {
        {"type_aliases",
         jst::fmt::format("using U64 = unsigned long long;\n"
                          "using Scalar = {};",
                          "float")},
    };

    switch (path) {
        case PolarizerPath::BYPASS:
            break;
        case PolarizerPath::XY_TO_LR:
            kernelName = kPolarizerXYtoLRKernelName;
            JST_CHECK(createKernel(kernelName, kPolarizerXYtoLRKernelSource, pieces));
            break;
        case PolarizerPath::XY_TO_X:
            kernelName = kPolarizerXYtoXKernelName;
            JST_CHECK(createKernel(kernelName, kPolarizerXYtoXKernelSource, pieces));
            break;
        case PolarizerPath::XY_TO_Y:
            kernelName = kPolarizerXYtoYKernelName;
            JST_CHECK(createKernel(kernelName, kPolarizerXYtoYKernelSource, pieces));
            break;
    }

    kernelCreated = true;

    return Result::SUCCESS;
}

Result PolarizerImplNativeCuda::computeSubmit(const cudaStream_t& stream) {
    if (bypass) {
        return Result::SUCCESS;
    }

    const auto* inputBase = static_cast<const std::uint8_t*>(inputTensor.buffer().data());
    auto* outputBase = static_cast<std::uint8_t*>(outputTensor.buffer().data());
    if (!inputBase || !outputBase) {
        JST_ERROR("[MODULE_POLARIZER_NATIVE_CUDA] Missing input or output device buffer.");
        return Result::ERROR;
    }

    const void* inputData = inputBase + inputTensor.offsetBytes();
    void* outputData = outputBase + outputTensor.offsetBytes();

    JST_CUDA_CHECK(cudaMemsetAsync(outputData, 0, outputTensor.sizeBytes(), stream), [&] {
        JST_ERROR("[MODULE_POLARIZER_NATIVE_CUDA] Failed to clear the output buffer: {}.", err);
    });

    void* inputArgument = const_cast<void*>(inputData);
    void* arguments[] = {
        &inputArgument,
        &outputData,
        &outputWorkItemCount,
    };

    const Extent3D<U64> block = {launchBlockSize, 1, 1};
    const Extent3D<U64> grid = {
        outputWorkItemCount / launchBlockSize +
            (outputWorkItemCount % launchBlockSize != 0),
        1,
        1,
    };

    JST_CHECK(scheduleKernel(kernelName, stream, grid, block, arguments));

    return Result::SUCCESS;
}

Result PolarizerImplNativeCuda::computeDeinitialize() {
    if (kernelCreated) {
        JST_CHECK(destroyKernel(kernelName));
    }

    kernelCreated = false;
    kernelName.clear();

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(PolarizerImplNativeCuda, DeviceType::CUDA, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
