#include <jetstream/backend/devices/cuda/helpers.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>
#include <jetstream/runtime_context_native_cuda.hh>
#include <jetstream/scheduler_context.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

namespace {

constexpr const char* kDetector4PolKernelSource = R"(
extern "C" __global__ void detector_4pol(const float* input,
                                         float* output,
                                         unsigned long long sampleCount,
                                         unsigned long long integrationRate) {
    const unsigned long long index =
        (static_cast<unsigned long long>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (index >= sampleCount) {
        return;
    }

    const unsigned long long inputOffset = index * 4;
    const float xr = input[inputOffset + 0];
    const float xi = input[inputOffset + 1];
    const float yr = input[inputOffset + 2];
    const float yi = input[inputOffset + 3];

    const float xx = (xr * xr) + (xi * xi);
    const float yy = (yr * yr) + (yi * yi);
    const float zr = (xr * yr) + (xi * yi);
    const float zi = (xi * yr) - (xr * yi);

    const unsigned long long outputOffset = (index / integrationRate) * 4;
    atomicAdd(output + outputOffset + 0, xx);
    atomicAdd(output + outputOffset + 1, yy);
    atomicAdd(output + outputOffset + 2, zr);
    atomicAdd(output + outputOffset + 3, zi);
}
)";

constexpr const char* kDetector1PolKernelSource = R"(
extern "C" __global__ void detector_1pol(const float* input,
                                         float* output,
                                         unsigned long long sampleCount,
                                         unsigned long long integrationRate) {
    const unsigned long long index =
        (static_cast<unsigned long long>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (index >= sampleCount) {
        return;
    }

    const unsigned long long inputOffset = index * 4;
    const float xr = input[inputOffset + 0];
    const float xi = input[inputOffset + 1];
    const float yr = input[inputOffset + 2];
    const float yi = input[inputOffset + 3];

    const float xx = (xr * xr) + (xi * xi);
    const float yy = (yr * yr) + (yi * yi);

    atomicAdd(output + (index / integrationRate), xx + yy);
}
)";

constexpr const char* kDetector4PolKernelName = "detector_4pol";
constexpr const char* kDetector1PolKernelName = "detector_1pol";

}  // namespace

struct DetectorImplNativeCuda : public DetectorImpl,
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
};

Result DetectorImplNativeCuda::create() {
    const Tensor& input = inputs().at("buffer").tensor;

    if (input.dtype() != DataType::CF32) {
        JST_ERROR("[MODULE_DETECTOR_NATIVE_CUDA] Unsupported input data type '{}'. Expected CF32.",
                  input.dtype());
        return Result::ERROR;
    }

    JST_CHECK(DetectorImpl::create());

    return Result::SUCCESS;
}

Result DetectorImplNativeCuda::computeInitialize() {
    if (numberOfOutputPolarizations == 4) {
        kernelName = kDetector4PolKernelName;
        JST_CHECK(createKernel(kDetector4PolKernelName, kDetector4PolKernelSource));
    } else if (numberOfOutputPolarizations == 1) {
        kernelName = kDetector1PolKernelName;
        JST_CHECK(createKernel(kDetector1PolKernelName, kDetector1PolKernelSource));
    } else {
        JST_ERROR("[MODULE_DETECTOR_NATIVE_CUDA] Unsupported number of output polarizations {}.",
                  numberOfOutputPolarizations);
        return Result::ERROR;
    }

    kernelCreated = true;

    return Result::SUCCESS;
}

Result DetectorImplNativeCuda::computeSubmit(const cudaStream_t& stream) {
    JST_CUDA_CHECK(cudaMemsetAsync(outputTensor.data(), 0, outputTensor.sizeBytes(), stream), [&] {
        JST_ERROR("[MODULE_DETECTOR_NATIVE_CUDA] Failed to clear the detector output buffer: {}.", err);
    });

    const void* inputData = inputTensor.data();
    void* outputData = outputTensor.data();

    void* inputArgument = const_cast<void*>(inputData);
    void* arguments[] = {
        &inputArgument,
        &outputData,
        &inputSampleCount,
        &integrationRate,
    };

    const Extent3D<U64> block = {blockSize, 1, 1};
    const Extent3D<U64> grid = {(inputSampleCount + blockSize - 1) / blockSize, 1, 1};

    JST_CHECK(scheduleKernel(kernelName, stream, grid, block, arguments));

    if (inputTensor.hasAttribute("timestamp")) {
        JST_CHECK(outputTensor.setAttribute("timestamp", inputTensor.attribute("timestamp")));
    }

    return Result::SUCCESS;
}

Result DetectorImplNativeCuda::computeDeinitialize() {
    if (kernelCreated) {
        JST_CHECK(destroyKernel(kernelName));
    }

    kernelCreated = false;
    kernelName.clear();

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(DetectorImplNativeCuda, DeviceType::CUDA, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
