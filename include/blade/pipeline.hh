#ifndef BLADE_PIPELINE_HH
#define BLADE_PIPELINE_HH

#include <span>
#include <string>
#include <memory>
#include <vector>

#include "blade/logger.hh"
#include "blade/module.hh"

namespace Blade {

class BLADE_API Pipeline {
 public:
    Pipeline();
    virtual ~Pipeline();

    Result synchronize();
    bool isSynchronized();

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

    template<typename T>
    Result copy(Vector<Device::CPU, T>& dst,
                const Vector<Device::CUDA | Device::CPU, T>& src) {
        return Memory::Copy(dst, src, this->stream);
    }

    template<typename T>
    Result copy(Vector<Device::CUDA, T>& dst,
                const Vector<Device::CUDA | Device::CPU, T>& src) {
        return Memory::Copy(dst, src, this->stream);
    }

    template<typename T>
    Result copy(Vector<Device::CUDA | Device::CPU, T>& dst,
                const Vector<Device::CUDA, T>& src) {
        return Memory::Copy(dst, src, this->stream);
    }

    template<typename T>
    Result copy(Vector<Device::CUDA | Device::CPU, T>& dst,
                const Vector<Device::CPU, T>& src) {
        return Memory::Copy(dst, src, this->stream);
    }

    template<typename T>
    Result copy(Vector<Device::CUDA | Device::CPU, T>& dst,
                const Vector<Device::CUDA | Device::CPU, T>& src) {
        return Memory::Copy(dst, src, this->stream);
    }

    template<typename DT, typename ST>
    Result copy2D(Vector<Device::CUDA, DT>& dst,
                   const U64& dst_pitch,
                   const U64& dst_pad,
                   const Vector<Device::CUDA, ST>& src,
                   const U64& src_pitch,
                   const U64& src_pad,
                   const U64& width,
                   const U64& height) {
        return Memory::Copy2D(dst, dst_pitch, src_pad, src, src_pitch, 
            src_pad, width, height, this->stream);
    }

    template<typename DT, typename ST>
    Result copy2D(Vector<Device::CUDA, DT>& dst,
                   const U64& dst_pitch,
                   const U64& dst_pad,
                   const Vector<Device::CPU, ST>& src,
                   const U64& src_pitch,
                   const U64& src_pad,
                   const U64& width,
                   const U64& height) {
        return Memory::Copy2D(dst, dst_pitch, src_pad, src, src_pitch, 
            src_pad, width, height, this->stream);
    }

    template<typename DT, typename ST>
    Result copy2D(Vector<Device::CPU, DT>& dst,
                   const U64& dst_pitch,
                   const U64& dst_pad,
                   const Vector<Device::CPU, ST>& src,
                   const U64& src_pitch,
                   const U64& src_pad,
                   const U64& width,
                   const U64& height) {
        return Memory::Copy2D(dst, dst_pitch, src_pad, src, src_pitch, 
            src_pad, width, height, this->stream);
    }

    template<typename DT, typename ST>
    Result copy2D(Vector<Device::CPU, DT>& dst,
                   const U64& dst_pitch,
                   const U64& dst_pad,
                   const Vector<Device::CUDA, ST>& src,
                   const U64& src_pitch,
                   const U64& src_pad,
                   const U64& width,
                   const U64& height) {
        return Memory::Copy2D(dst, dst_pitch, src_pad, src, src_pitch, 
            src_pad, width, height, this->stream);
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
