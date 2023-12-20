#ifndef BLADE_MEMORY_PROFILER_HH
#define BLADE_MEMORY_PROFILER_HH

#include "blade/macros.hh"
#include "blade/memory/types.hh"

namespace Blade {

class BLADE_API Profiler {
 public:
    struct Capture {
        U64 allocatedCudaMemory;
        U64 deallocatedCudaMemory;
        U64 allocatedCudaTensors;
        U64 deallocatedCudaTensors;

        U64 allocatedCpuMemory;
        U64 deallocatedCpuMemory;
        U64 allocatedCpuTensors;
        U64 deallocatedCpuTensors;

        U64 allocatedUnifiedMemory;
        U64 deallocatedUnifiedMemory;
        U64 allocatedUnifiedTensors;
        U64 deallocatedUnifiedTensors;
    };

    Profiler(Profiler const&) = delete;
    void operator=(Profiler const&) = delete;

    static Profiler& GetInstance();

    static void StartCapture() {
        return GetInstance().startCapture();
    }

    static Capture StopCapture() {
        return GetInstance().stopCapture();
    }

    static void PrintCapture() {
        return GetInstance().printCapture();
    }

    static bool IsCapturing() {
        return GetInstance().isCapturing();
    }

    static void RegisterCudaAllocation(const U64& byteSize) {
        return GetInstance().registerCudaAllocation(byteSize);
    }

    static void RegisterCpuAllocation(const U64& byteSize) {
        return GetInstance().registerCpuAllocation(byteSize);
    }

    static void RegisterUnifiedAllocation(const U64& byteSize) {
        return GetInstance().registerUnifiedAllocation(byteSize);
    }

    static void RegisterCudaDeallocation(const U64& byteSize) {
        return GetInstance().registerCudaDeallocation(byteSize);
    }

    static void RegisterCpuDeallocation(const U64& byteSize) {
        return GetInstance().registerCpuDeallocation(byteSize);
    }

    static void RegisterUnifiedDeallocation(const U64& byteSize) {
        return GetInstance().registerUnifiedDeallocation(byteSize);
    }

 private:
    Capture capture;
    bool _isCapturing;

    Profiler() : _isCapturing(false) {}

    void startCapture();
    Capture stopCapture();
    void printCapture();

    bool isCapturing();
    void registerCudaAllocation(const U64& byteSize);
    void registerCpuAllocation(const U64& byteSize);
    void registerUnifiedAllocation(const U64& byteSize);
    void registerCudaDeallocation(const U64& byteSize);
    void registerCpuDeallocation(const U64& byteSize);
    void registerUnifiedDeallocation(const U64& byteSize);
};

}  // namespace Blade

#endif
