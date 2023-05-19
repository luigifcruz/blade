#define BL_LOG_DOMAIN "P::MEMORY"

#include "blade/memory/profiler.hh"

namespace Blade::Memory {

Profiler& Profiler::GetInstance() {
    static Profiler instance;
    return instance;
}

void Profiler::startCapture() {
    if (isCapturing()) {
        BL_FATAL("Can't start capture because there is an ongoing capture.");
        BL_CHECK_THROW(Result::ERROR);
    }
    BL_TRACE("Starting memory capture.");
    capture = Profiler::Capture({0});
    _isCapturing = true;
}

Profiler::Capture Profiler::stopCapture() {
    if (!isCapturing()) {
        BL_FATAL("Can't stop capture because there is no ongoing capture.");
        BL_CHECK_THROW(Result::ERROR);
    }
    BL_TRACE("Stopping memory capture.");
    _isCapturing = false;
    return capture;
}

void Profiler::printCapture() {
    BL_INFO("======== Memory Profile Capture ========");
    BL_INFO("--- CUDA -------------------------------")
    BL_INFO("Allocated: {}", ReadableBytes(capture.allocatedCudaMemory));
    BL_INFO("Deallocated: {}", ReadableBytes(capture.deallocatedCudaMemory));
    BL_INFO("Tensors Allocated: {}", capture.allocatedCudaTensors);
    BL_INFO("Tensors Deallocated: {}", capture.deallocatedCudaTensors);
    BL_INFO("--- CPU --------------------------------")
    BL_INFO("Allocated: {}", ReadableBytes(capture.allocatedCpuMemory));
    BL_INFO("Deallocated: {}", ReadableBytes(capture.deallocatedCpuMemory));
    BL_INFO("Tensors Allocated: {}", capture.allocatedCpuTensors);
    BL_INFO("Tensors Deallocated: {}", capture.deallocatedCpuTensors);
    BL_INFO("--- UNIFIED ----------------------------")
    BL_INFO("Allocated: {}", ReadableBytes(capture.allocatedUnifiedMemory));
    BL_INFO("Deallocated: {}", ReadableBytes(capture.deallocatedUnifiedMemory));
    BL_INFO("Tensors Allocated: {}", capture.allocatedUnifiedTensors);
    BL_INFO("Tensors Deallocated: {}", capture.deallocatedUnifiedTensors);
    BL_INFO("========================================");
}

bool Profiler::isCapturing() {
    return _isCapturing;
}

void Profiler::registerCudaAllocation(const U64& byteSize) {
    if (isCapturing()) {
        capture.allocatedCudaMemory += byteSize;
        capture.allocatedCudaTensors += 1;       
    }
}

void Profiler::registerCpuAllocation(const U64& byteSize) {
    if (isCapturing()) {
        capture.allocatedCpuMemory += byteSize;
        capture.allocatedCpuTensors += 1;       
    }
}

void Profiler::registerUnifiedAllocation(const U64& byteSize) {
    if (isCapturing()) {
        capture.allocatedUnifiedMemory += byteSize;
        capture.allocatedUnifiedTensors += 1;       
    }
}

void Profiler::registerCudaDeallocation(const U64& byteSize) {
    if (isCapturing()) {
        capture.deallocatedCudaMemory += byteSize;
        capture.deallocatedCudaTensors += 1;       
    }
}

void Profiler::registerCpuDeallocation(const U64& byteSize) {
    if (isCapturing()) {
        capture.deallocatedCpuMemory += byteSize;
        capture.deallocatedCpuTensors += 1;       
    }
}

void Profiler::registerUnifiedDeallocation(const U64& byteSize) {
    if (isCapturing()) {
        capture.deallocatedUnifiedMemory += byteSize;
        capture.deallocatedUnifiedTensors += 1;       
    }
}

}  // namespace Blade::Memory