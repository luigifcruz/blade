#include <jetstream/backend/devices/cuda/helpers.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>
#include <jetstream/runtime_context_native_cuda.hh>
#include <jetstream/scheduler_context.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

namespace {

constexpr const char* kStackerKernelSource = R"(
<<<type_aliases>>>
<<<kernel_constants>>>

struct alignas(2 * sizeof(InputScalar)) Complex {
    InputScalar real;
    InputScalar imag;
};

extern "C" __global__ void stacker(const Complex* input,
                                         Complex* output,
                                   const U64 input_size,
                                   const U64 stackIndex) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < input_size) {
        const U64 oid = (tid/WIDTH_IN)*(WIDTH_OUT) + (stackIndex*WIDTH_IN) + (tid%WIDTH_IN);
        output[oid] = input[tid];
    }
}
)";

constexpr const char* kStackerKernelName  = "stacker";

}  // namespace

struct StackerImplNativeCuda : public StackerImpl,
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
    bool kernelNotCopy = false;
    U64 width, widthByteSize;
    U64 height;
    U64 inputSize;
    U64 stackIndex;
};

Result StackerImplNativeCuda::create() {
    const Tensor& input = inputs().at("buffer").tensor;

    if (input.dtype() != DataType::CF32) {
        JST_ERROR("[MODULE_STACKER_NATIVE_CUDA] Unsupported input data type '{}'. Expected CF32.",
                  input.dtype());
        return Result::ERROR;
    }

    stackIndex = 0;
    width = 1;
    for (U64 i = axis; i < input.rank(); i++) {
        width *= input.shape()[i];
    }
    widthByteSize = width * sizeof(input.dtype());
    height = 1;
    for (U64 i = 0; i < axis; i++) {
        height *= input.shape()[i];
    }
    inputSize = input.size();
    JST_DEBUG("[MODULE_STACKER_NATIVE_CUDA] Height of {} and width of {} elements ({} bytes).",
              height, width, widthByteSize)
    kernelNotCopy = width < copySizeThreshold;
    JST_DEBUG("[MODULE_STACKER_NATIVE_CUDA] Stacking with {}.", kernelNotCopy ? "kernel" : "CUDA memcopy");

    JST_CHECK(StackerImpl::create());

    return Result::SUCCESS;
}

Result StackerImplNativeCuda::computeInitialize() {
    if (kernelNotCopy) {
        const std::string scalarType = [&]() -> std::string {
            switch (inputTensor.dtype()) {
                case DataType::CF32: return "float";
                default: return "";
            }
        }();
        if (scalarType.empty()) {
            JST_ERROR("[MODULE_STACKER_NATIVE_CUDA] Unsupported input data type '{}'. Expected CF32.",
                    inputTensor.dtype());
            return Result::ERROR;
        }
        
        const std::unordered_map<std::string, std::string> pieces = {
            {"type_aliases",
            jst::fmt::format("using U64 = unsigned long long;\n"
                            "using InputScalar = {};",
                            scalarType)},
            {"kernel_constants",
            jst::fmt::format("static constexpr int WIDTH_IN = {};\n"
                            "static constexpr int WIDTH_OUT = {};",
                            width,
                            width*ratio)},
        };
        JST_CHECK(createKernel(kStackerKernelName, kStackerKernelSource, pieces));
        kernelCreated = true;
    }
    return Result::SUCCESS;
}

Result StackerImplNativeCuda::computeSubmit(const cudaStream_t& stream) {
    if (stackIndex == 0) {
        JST_CUDA_CHECK(cudaMemsetAsync(outputTensor.data(), 0, outputTensor.sizeBytes(), stream), [&] {
            JST_ERROR("[MODULE_STACKER_NATIVE_CUDA] Failed to clear the output buffer: {}.", err);
        });
    }
    const void* inputData = inputTensor.data();
    void* outputData = outputTensor.data();
    if (kernelNotCopy) {
        void* inputArgument = const_cast<void*>(inputData);
        void* arguments[] = {
            &inputArgument,
            &outputData,
            (void*)&inputSize,
            (void*)&stackIndex
        };
    
        const Extent3D<U64> block = {blockSize, 1, 1};
        const Extent3D<U64> grid = {(inputTensor.size() + blockSize - 1) / blockSize, 1, 1};
    
        JST_CHECK(scheduleKernel(kStackerKernelName, stream, grid, block, arguments));
    } else {
        
        JST_CUDA_CHECK(
            cudaMemcpy2DAsync(
                ((uint8_t*)outputData)+(widthByteSize * stackIndex),
                widthByteSize * ratio,
                ((uint8_t*)inputData)+0,
                widthByteSize,
                widthByteSize,
                height,
                cudaMemcpyDeviceToDevice,
                stream
            ), [&] {
            JST_ERROR("[MODULE_STACKER_NATIVE_CUDA] Failed to copy to the output buffer: {}.", err);
        });
    }

    if (inputTensor.hasAttribute("timestamp")) {
        JST_CHECK(outputTensor.setAttribute("timestamp", inputTensor.attribute("timestamp")));
    }
    stackIndex = (stackIndex + 1)%this->ratio;

    return Result::SUCCESS;
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
