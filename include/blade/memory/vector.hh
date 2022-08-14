#ifndef BLADE_MEMORY_VECTOR_HH
#define BLADE_MEMORY_VECTOR_HH

#include "blade/memory/types.hh"

namespace Blade {

template<Device I, typename T> class Vector;

template<typename T>
class VectorImpl {
 public:
    VectorImpl()
             : container(),
               managed(false) {}
    explicit VectorImpl(const U64& size)
             : container(),
               managed(true) {
        this->resize(size);
    }
    explicit VectorImpl(const std::span<T>& other)
             : container(other),
               managed(false) {}
    explicit VectorImpl(T* ptr, const U64& size)
             : container(ptr, size),
               managed(false) {}
    explicit VectorImpl(void* ptr, const U64& size)
             : container(static_cast<T*>(ptr), size),
               managed(false) {}

    VectorImpl(const VectorImpl&) = delete;
    VectorImpl& operator=(const VectorImpl&) = delete;

    virtual ~VectorImpl() {}

    constexpr T* data() const noexcept {
        return container.data();
    }

    constexpr U64 size() const noexcept {
        return container.size();
    }

    constexpr U64 size_bytes() const noexcept {
        return container.size_bytes();
    }

    [[nodiscard]] constexpr bool empty() const noexcept {
        return container.empty();
    }

    constexpr T& operator[](U64 idx) const {
        return container[idx];
    }

    // TODO: Implement iterator.
    constexpr const std::span<T>& getUnderlying() const {
        return container;
    }

    const Result link(const VectorImpl<T>& src) {
        if (src.empty()) {
            BL_FATAL("Source can't be empty while linking.");
            return Result::ERROR;
        }

        this->managed = false;
        this->container = src.getUnderlying();

        return Result::SUCCESS;
    }

    virtual const Result resize(const U64& size) = 0;

 protected:
    std::span<T> container;
    bool managed;
};

}  // namespace Blade

#endif
