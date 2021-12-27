#ifndef BLADE_PIPELINE_HH
#define BLADE_PIPELINE_HH

#include <span>
#include <string>
#include <memory>

#include "blade/logger.hh"
#include "blade/module.hh"

namespace Blade {

class BLADE_API Pipeline {
 public:
    Pipeline();
    virtual ~Pipeline();

    Result synchronize();
    bool isSyncronized();

 protected:
    template<typename T>
    void connect(std::shared_ptr<T>& module,
                 const typename T::Config& config,
                 const typename T::Input& input) {
        module = std::make_unique<T>(config, input);
        this->modules.push_back(module);
    }

    Result compute();

    template<typename T>
    Result copy(Vector<Device::CUDA, T>& dst,
                const Vector<Device::CUDA, T>& src) {
        return Memory::Copy(dst, src, this->stream);
    }

    template<typename T>
    Result copy(Vector<Device::CUDA, T>& dst,
                const Vector<Device::CPU, T>& src) {
        return Memory::Copy(dst, src, this->stream);
    }

    template<typename T>
    Result copy(Vector<Device::CPU, T>& dst,
                const Vector<Device::CPU, T>& src) {
        return Memory::Copy(dst, src);
    }

    template<typename T>
    Result copy(Vector<Device::CPU, T>& dst,
                const Vector<Device::CUDA, T>& src) {
        return Memory::Copy(dst, src, this->stream);
    }

 private:
    enum State : uint8_t {
        IDLE,
        CACHED,
        GRAPH,
    };

    State state;
    cudaGraph_t graph;
    cudaStream_t stream;
    cudaGraphExec_t instance;
    std::vector<std::shared_ptr<Module>> modules;
};

}  // namespace Blade

#endif
