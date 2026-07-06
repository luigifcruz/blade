#include <jetstream/backend/devices/cuda/helpers.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>
#include <jetstream/runtime_context_native_cuda.hh>
#include <jetstream/scheduler_context.hh>

#include "module_impl.hh"

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
                                                 U64 inputSize,
                                                 U64 outputSize) {
    const int tid = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

    assert(inputSize == outputSize);

    if (tid < inputSize) {
        // The complex multiplication below can be simplified because
        // the real part of the phasor is 0.0. Boring implementation:
        // const IT yPol90 = cuCmulf(yPol, make_cuFloatComplex(0.0, 1.0));

        const Complex xPol = input[tid + 0];
        const Complex yPol = input[tid + 1];

        const Complex yPol90(-yPol.imag, +yPol.real);
        
        output[tid + 0] = add(xPol, yPol90);
        output[tid + 1] = sub(xPol, yPol90);
    }
}
)";

constexpr const char* kPolarizerXYtoXKernelSource = R"(
<<<type_aliases>>>

struct alignas(2 * sizeof(Scalar)) Complex {
    Scalar real;
    Scalar imag;
};

extern "C" __global__ __global__ void polarizer_xy_x(const Complex* input,
                                                           Complex* output,
                                                           U64 inputSize,
                                                           U64 outputSize) {
    const int tid = (blockIdx.x * blockDim.x + threadIdx.x);

    if (tid < outputSize) {
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

extern "C" __global__ __global__ void polarizer_xy_y(const Complex* input,
                                                           Complex* output,
                                                           U64 inputSize,
                                                           U64 outputSize) {
    const int tid = (blockIdx.x * blockDim.x + threadIdx.x);

    if (tid < outputSize) {
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
    Result create() final;
    Result computeInitialize() override;
    Result computeSubmit(const cudaStream_t& stream) override;
    Result computeDeinitialize() override;

 private:
    std::string kernelName;
    bool kernelCreated = false;
    U64 inputSize, outputSize;
};

Result PolarizerImplNativeCuda::create() {
    const Tensor& input = inputs().at("buffer").tensor;

    if (input.dtype() != DataType::CF32) {
        JST_ERROR("[MODULE_POLARIZER_NATIVE_CUDA] Unsupported input data type '{}'. Expected CF32.",
                  input.dtype());
        return Result::ERROR;
    }

    JST_CHECK(PolarizerImpl::create());
    inputSize = input.size();
    outputSize = outputs().at("buffer").tensor.size();

    return Result::SUCCESS;
}

Result PolarizerImplNativeCuda::computeInitialize() {
    const std::string scalarType = [&]() -> std::string {
        switch (inputTensor.dtype()) {
            case DataType::CF32: return "float";
            default: return "";
        }
    }();
    if (scalarType.empty()) {
        JST_ERROR("[MODULE_POLARIZER_NATIVE_CUDA] Unsupported input data type '{}'. Expected CF32.",
                inputTensor.dtype());
        return Result::ERROR;
    }
    
    const std::unordered_map<std::string, std::string> pieces = {
        {"type_aliases",
        jst::fmt::format("using U64 = unsigned long long;\n"
                        "using Scalar = {};",
                        scalarType)},
    };

    if (outputPolarization == "lr") {
        kernelName = kPolarizerXYtoLRKernelName;
        JST_CHECK(createKernel(kPolarizerXYtoLRKernelName, kPolarizerXYtoLRKernelSource, pieces));
    } else if (outputPolarization == "x") {
        kernelName = kPolarizerXYtoXKernelName;
        JST_CHECK(createKernel(kPolarizerXYtoXKernelName, kPolarizerXYtoXKernelSource, pieces));
    } else if (outputPolarization == "y") {
        kernelName = kPolarizerXYtoYKernelName;
        JST_CHECK(createKernel(kPolarizerXYtoYKernelName, kPolarizerXYtoYKernelSource, pieces));
    } else {
        JST_ERROR("[MODULE_POLARIZER_NATIVE_CUDA] Unsupported output polarization {}.",
                  outputPolarization);
        return Result::ERROR;
    }

    kernelCreated = true;

    return Result::SUCCESS;
}

Result PolarizerImplNativeCuda::computeSubmit(const cudaStream_t& stream) {
    JST_CUDA_CHECK(cudaMemsetAsync(outputTensor.data(), 0, outputTensor.sizeBytes(), stream), [&] {
        JST_ERROR("[MODULE_POLARIZER_NATIVE_CUDA] Failed to clear the output buffer: {}.", err);
    });

    const void* inputData = inputTensor.data();
    void* outputData = outputTensor.data();

    void* inputArgument = const_cast<void*>(inputData);
    void* arguments[] = {
        &inputArgument,
        &outputData,
        (void*)&inputSize,
        (void*)&outputSize
    };

    const Extent3D<U64> block = {blockSize, 1, 1};
    const Extent3D<U64> grid = {(inputTensor.size() + blockSize - 1) / blockSize, 1, 1};

    JST_CHECK(scheduleKernel(kernelName, stream, grid, block, arguments));

    if (inputTensor.hasAttribute("timestamp")) {
        JST_CHECK(outputTensor.setAttribute("timestamp", inputTensor.attribute("timestamp")));
    }

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
