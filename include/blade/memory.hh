#ifndef BLADE_MEMORY_HH
#define BLADE_MEMORY_HH

#include <vector>

#include "blade/common.hh"
#include "blade/logger.hh"

namespace Blade {

class BLADE_API Memory {
 protected:
    Memory() {}

    template<class T>
    class CustomVector {
     public:
        CustomVector()
                 : container(),
                   managed(false) {}
        explicit CustomVector(const std::span<T>& other)
                 : container(other),
                   managed(false) {}
        explicit CustomVector(const T* ptr, const std::size_t& size)
                 : container(ptr, size),
                   managed(false) {}

        CustomVector(const CustomVector&) = delete;
        CustomVector& operator=(const CustomVector&) = delete;

        constexpr T* data() const noexcept {
            return container.data();
        }

        constexpr std::size_t size() const noexcept {
            return container.size();
        }

        constexpr std::size_t size_bytes() const noexcept {
            return container.size_bytes();
        }

        [[nodiscard]] constexpr bool empty() const noexcept {
            return container.empty();
        }

        constexpr T& operator[](std::size_t idx) const {
            return container[idx];
        }

     protected:
        std::span<T> container;
        bool managed;
    };

    template<typename T>
    static Result Copy(CustomVector<T>& dst,
                       const CustomVector<T>& src,
                       const cudaMemcpyKind& kind,
                       const cudaStream_t& stream = 0) {
        if (dst.size() != src.size()) {
            BL_FATAL("Size mismatch between source and destination ({}, {}).",
                    src.size(), dst.size());
            return Result::ASSERTION_ERROR;
        }

        BL_CUDA_CHECK(cudaMemcpyAsync(dst.data(), src.data(), src.size_bytes(),
                    kind, stream), [&]{
            BL_FATAL("Can't copy data: {}", err);
        });

        return Result::SUCCESS;
    }

    static constexpr std::size_t ToMB(const std::size_t& size) {
        return size / 1e6;
    }

 public:
    struct Resources {
        std::size_t device = 0;
        std::size_t host = 0;
    };

    Memory(const Memory&) = delete;

    static Memory& Get() {
        static Memory instance;
        return instance;
    }

    static Resources GetResources() {
        return Get().resources;
    }

    static Result Reset() {
        Get().resources.device = 0;
        Get().resources.host = 0;

        return Result::SUCCESS;
    }

    static Result Report() {
        BL_INFO("=============================================");
        BL_INFO("Pipeline resources manager usage report:")
        BL_INFO("=============================================");
        BL_INFO("Memory usage:");
        BL_INFO("   Host:   {} MB", ToMB(Get().resources.host));
        BL_INFO("   Device: {} MB", ToMB(Get().resources.device));
        BL_INFO("=============================================");

        return Result::SUCCESS;
    }

    template<class T>
    class HostVector : public CustomVector<T> {
     public:
        using CustomVector<T>::CustomVector;

        explicit HostVector(const std::size_t& size) {
            BL_CHECK_THROW(this->allocate(size));
        }

        Result allocate(const std::size_t& size) {
            if (!this->container.empty() && !this->managed) {
                return Result::ERROR;
            }

            T* ptr;
            auto size_bytes = size * sizeof(T);

            BL_CUDA_CHECK(cudaMallocHost(&ptr, size_bytes), [&]{
                BL_FATAL("Failed to allocate host memory: {}", err);
            });

            Get().resources.host += size_bytes;
            this->container = std::span<T>(ptr, size);
            this->managed = true;

            return Result::SUCCESS;
        }

        ~HostVector() {
            if (!this->container.empty() && this->managed) {
                Get().resources.host -= this->container.size_bytes();
                if (cudaFreeHost(this->container.data()) != cudaSuccess) {
                    BL_FATAL("Failed to deallocate host memory.");
                };
            }
        }
    };

    template<class T>
    class DeviceVector : public CustomVector<T> {
     public:
        using CustomVector<T>::CustomVector;

        explicit DeviceVector(const std::size_t& size) {
            BL_CHECK_THROW(this->allocate(size));
        }

        Result allocate(const std::size_t& size) {
            if (!this->container.empty() && !this->managed) {
                return Result::ERROR;
            }

            T* ptr;
            auto size_bytes = size * sizeof(T);

            BL_CUDA_CHECK(cudaMalloc(&ptr, size_bytes), [&]{
                BL_FATAL("Failed to allocate device memory: {}", err);
            });

            Get().resources.device += size_bytes;
            this->container = std::span<T>(ptr, size);
            this->managed = true;

            return Result::SUCCESS;
        }

        ~DeviceVector() {
            if (!this->container.empty() && this->managed) {
                Get().resources.device -= this->container.size_bytes();
                if (cudaFree(this->container.data()) != cudaSuccess) {
                    BL_FATAL("Failed to deallocate device memory.");
                };
            }
        }
    };

    template<typename T>
    static Result Copy(DeviceVector<T>& dst,
                       const DeviceVector<T>& src,
                       const cudaStream_t& stream = 0) {
        return Get().Copy(dst, src, cudaMemcpyDeviceToDevice, stream);
    }

    template<typename T>
    static Result Copy(DeviceVector<T>& dst,
                       const HostVector<T>& src,
                       const cudaStream_t& stream = 0) {
        return Get().Copy(dst, src, cudaMemcpyHostToDevice, stream);
    }

    template<typename T>
    static Result Copy(HostVector<T>& dst,
                       const HostVector<T>& src,
                       const cudaStream_t& stream = 0) {
        return Get().Copy(dst, src, cudaMemcpyHostToHost, stream);
    }

    template<typename T>
    static Result Copy(HostVector<T>& dst,
                       const DeviceVector<T>& src,
                       const cudaStream_t& stream = 0) {
        return Get().Copy(dst, src, cudaMemcpyDeviceToHost, stream);
    }

    template<typename T>
    static Result Register(const std::span<T>& mem, const bool& readOnly = false) {
        unsigned int kind = cudaHostRegisterDefault;
        if (readOnly) {
            kind = cudaHostRegisterReadOnly;
        }

        BL_CUDA_CHECK(cudaHostRegister(mem.data(), mem.size_bytes(), kind), [&]{
            BL_FATAL("Failed to register host memory: {}", err);
        });

        Get().resources.host += mem.size_bytes();

        return Result::SUCCESS;
    }

 private:
    Resources resources;
};

}  // namespace Blade

#endif
