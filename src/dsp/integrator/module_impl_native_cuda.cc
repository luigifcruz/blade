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
    Result create() final;
    Result computeInitialize() override;
    Result computeSubmit(const cudaStream_t& stream) override;
    Result computeDeinitialize() override;

 private:
    std::string kernelName;
    bool kernelCreated = false;
    U64 componentCount = 1;
    U64 numberOfElements;
    U64 integratedElementCount;
    U64 blockIndex;
};

Result IntegratorImplNativeCuda::create() {
    const Tensor& input = inputs().at("buffer").tensor;

    if (
        input.dtype() != DataType::F32 &&
        input.dtype() != DataType::CF32 &&
        input.dtype() != DataType::CI8
    ) {
        JST_ERROR("[MODULE_INTEGRATOR_NATIVE_CUDA] Unsupported input data type '{}'. Expected F32, CF32, or CI8.",
                  input.dtype());
        return Result::ERROR;
    }

    JST_CHECK(IntegratorImpl::create());

    blockIndex = 0;
    if (bypass) {
        return Result::SUCCESS;
    }

    integratedElementCount = 1;
    for (U64 i = axis + 1; i < input.rank(); i++) {
        integratedElementCount *= input.shape()[i];
    }

    numberOfElements = input.size() / size;
    componentCount = IsDataTypeComplex(input.dtype()) ? 2 : 1;

    return Result::SUCCESS;
}

Result IntegratorImplNativeCuda::computeInitialize() {
    if (bypass) {
        return Result::SUCCESS;
    }

    const std::string scalarType = [&]() -> std::string {
        switch (inputTensor.dtype()) {
            case DataType::CF32: return "float";
            case DataType::F32: return "float";
            case DataType::CI8: return "signed char";
            default: return "";
        }
    }();
    if (scalarType.empty()) {
        JST_ERROR("[MODULE_INTEGRATOR_NATIVE_CUDA] Unsupported input data type '{}'. Expected F32, CF32, or CI8.",
                inputTensor.dtype());
        return Result::ERROR;
    }

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
    const Extent3D<U64> grid = {(numberOfElements + blockSize - 1) / blockSize, 1, 1};

    JST_CHECK(scheduleKernel(kIntegratorKernelName, stream, grid, block, arguments));

    if (inputTensor.hasAttribute("timestamp")) {
        JST_CHECK(outputTensor.setAttribute("timestamp", inputTensor.attribute("timestamp")));
    }
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
