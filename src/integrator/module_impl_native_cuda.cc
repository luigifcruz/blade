#include <jetstream/backend/devices/cuda/helpers.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>
#include <jetstream/runtime_context_native_cuda.hh>
#include <jetstream/scheduler_context.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

namespace {

constexpr const char* kIntegratorKernelSource = R"(
<<<type_aliases>>>
<<<kernel_constants>>>

extern "C" __global__ void integrator(const Scalar* input,
                                            Scalar* output,
                                      const U64 input_size) {
    const U64 tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < input_size) {
        Scalar accumulator[INTEGRATIONS] = {};

        for (U64 i = 0; i < INTEGRATION_SIZE; i++) {
            for (U64 j = 0; j < INTEGRATIONS; j++) {
                accumulator[j] += input[(tid * INTEGRATION_SIZE * INTEGRATIONS) + (i * INTEGRATIONS) + j];
            }
        }

        for (U64 j = 0; j < INTEGRATIONS; j++) {
            output[(tid * INTEGRATIONS) + j] += accumulator[j];
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
    Result create() final;
    Result computeInitialize() override;
    Result computeSubmit(const cudaStream_t& stream) override;
    Result computeDeinitialize() override;

 private:
    std::string kernelName;
    bool kernelCreated = false;
    bool dataIsComplex = false;
    U64 numberOfElements;
    U64 integratedElementCount;
    U64 blockIndex;
};

Result IntegratorImplNativeCuda::create() {
    const Tensor& input = inputs().at("buffer").tensor;

    if (
        input.dtype() != DataType::CF32 &&
        input.dtype() != DataType::CI8
    ) {
        JST_ERROR("[MODULE_INTEGRATOR_NATIVE_CUDA] Unsupported input data type '{}'. Expected CF32 or CI8.",
                  input.dtype());
        return Result::ERROR;
    }

    blockIndex = 0;
    integratedElementCount = 1;
    for (U64 i = axis+1; i < input.rank(); i++) {
        JST_DEBUG("*= axis#{}", axis);
        integratedElementCount *= input.shape()[i];
    }

    numberOfElements = input.size() / integratedElementCount / size;
    dataIsComplex = true;

    JST_CHECK(IntegratorImpl::create());

    return Result::SUCCESS;
}

Result IntegratorImplNativeCuda::computeInitialize() {
    const std::string scalarType = [&]() -> std::string {
        switch (inputTensor.dtype()) {
            case DataType::CF32: return "float";
            case DataType::F32: return "float";
            case DataType::CI8: return "signed char";
            default: return "";
        }
    }();
    if (scalarType.empty()) {
        JST_ERROR("[MODULE_INTEGRATOR_NATIVE_CUDA] Unsupported input data type '{}'. Expected CF32.",
                inputTensor.dtype());
        return Result::ERROR;
    }
    
    const std::unordered_map<std::string, std::string> pieces = {
        {"type_aliases",
        jst::fmt::format("using U64 = unsigned long long;\n"
                        "using Scalar = {};",
                        scalarType)},
        {"kernel_constants",
        jst::fmt::format("static constexpr int INTEGRATION_SIZE = {};\n"
                        "static constexpr int INTEGRATIONS = {};",
                        size * (dataIsComplex ? 2 : 1),
                        integratedElementCount * (dataIsComplex ? 2 : 1))},
    };
    JST_CHECK(createKernel(kIntegratorKernelName, kIntegratorKernelSource, pieces));
    kernelCreated = true;
    return Result::SUCCESS;
}

Result IntegratorImplNativeCuda::computeSubmit(const cudaStream_t& stream) {
    if (blockIndex == 0) {
        JST_CUDA_CHECK(cudaMemsetAsync(outputTensor.data(), 0, outputTensor.sizeBytes(), stream), [&] {
            JST_ERROR("[MODULE_INTEGRATOR_NATIVE_CUDA] Failed to clear the output buffer: {}.", err);
        });
    }
    const void* inputData = inputTensor.data();
    void* outputData = outputTensor.data();

    void* inputArgument = const_cast<void*>(inputData);
    void* arguments[] = {
        &inputArgument,
        &outputData,
        (void*)&numberOfElements
    };

    const Extent3D<U64> block = {blockSize, 1, 1};
    const Extent3D<U64> grid = {(inputTensor.size() + blockSize - 1) / blockSize, 1, 1};

    JST_CHECK(scheduleKernel(kIntegratorKernelName, stream, grid, block, arguments));

    if (inputTensor.hasAttribute("timestamp")) {
        JST_CHECK(outputTensor.setAttribute("timestamp", inputTensor.attribute("timestamp")));
    }
    blockIndex = (blockIndex + 1)%rate;

    return Result::SUCCESS;
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
