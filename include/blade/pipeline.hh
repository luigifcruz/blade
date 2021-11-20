#ifndef BLADE_PIPELINE_HH
#define BLADE_PIPELINE_HH

#include <span>
#include <string>
#include <memory>
#include <map>

#include "blade/common.hh"
#include "blade/logger.hh"
#include "blade/module.hh"


namespace Blade {

class BLADE_API Pipeline {
 public:
    Pipeline() : state(State::IDLE) {
        BL_CUDA_CHECK_THROW(cudaStreamCreateWithFlags(&this->stream,
                cudaStreamNonBlocking), [&]{
            BL_FATAL("Failed to create stream for CUDA steam: {}", err);
        });
    }

    virtual ~Pipeline() {
        this->synchronize();
        if (this->state == State::GRAPH) {
            cudaGraphDestroy(this->graph);
        }
        cudaStreamDestroy(this->stream);
    }

    Result synchronize() {
        BL_CUDA_CHECK(cudaStreamSynchronize(this->stream), [&]{
            BL_FATAL("Failed to synchronize stream: {}", err);
        });
        return Result::SUCCESS;
    }

    bool isSyncronized() {
        return cudaStreamQuery(this->stream) == cudaSuccess;
    }

 protected:
    template<typename T>
    void connect(std::shared_ptr<T>& module,
                 const std::string& moduleName,
                 const typename T::Config& config,
                 const typename T::Input& input) {
        module = std::make_unique<T>(config, input);
        this->modules.insert({moduleName, module});
    }

    Result compute() {
        for (auto& [name, module] : this->modules) {
            BL_CHECK(module->preprocess());
        }

        switch (state) {
            case State::GRAPH:
                BL_CUDA_CHECK(cudaGraphLaunch(this->instance, this->stream), [&]{
                    BL_FATAL("Failed launch CUDA graph: {}", err);
                });
                break;
            case State::CACHED:
                BL_DEBUG("Creating CUDA Graph.");
                BL_CUDA_CHECK(cudaStreamBeginCapture(this->stream,
                    cudaStreamCaptureModeGlobal), [&]{
                    BL_FATAL("Failed to begin the capture of CUDA Graph: {}", err);
                });

                for (auto& [name, module] : this->modules) {
                    BL_CHECK(module->process());
                }

                BL_CUDA_CHECK(cudaStreamEndCapture(this->stream, &this->graph), [&]{
                    BL_FATAL("Failed to end the capture of CUDA Graph: {}", err);
                });

                BL_CUDA_CHECK(cudaGraphInstantiate(&this->instance, this->graph,
                        NULL, NULL, 0), [&]{
                    BL_FATAL("Failed to instantiate CUDA Graph: {}", err);
                });

                this->state = State::GRAPH;
                break;
            case State::IDLE:
                BL_DEBUG("Caching kernels ahead of CUDA Graph instantiation.");
                for (auto& [name, module] : this->modules) {
                    BL_CHECK(module->process());
                }
                this->state = State::CACHED;
                break;
            default:
                BL_FATAL("Internal error.");
                return Result::ERROR;
        }

        BL_CUDA_CHECK_KERNEL([&]{
            BL_FATAL("Failed to process: {}", err);
            return Result::CUDA_ERROR;
        });

        return Result::SUCCESS;
    }

    template<typename T>
    Result copy(Memory::DeviceVector<T>& dst,
                const Memory::DeviceVector<T>& src) {
        return Memory::Copy(dst, src, this->stream);
    }

    template<typename T>
    Result copy(Memory::DeviceVector<T>& dst,
                const Memory::HostVector<T>& src) {
        return Memory::Copy(dst, src, this->stream);
    }

    template<typename T>
    Result copy(Memory::HostVector<T>& dst,
                const Memory::HostVector<T>& src) {
        return Memory::Copy(dst, src, this->stream);
    }

    template<typename T>
    Result copy(Memory::HostVector<T>& dst,
                const Memory::DeviceVector<T>& src) {
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
    std::map<std::string, std::shared_ptr<Module>> modules;
};

}  // namespace Blade

#endif
