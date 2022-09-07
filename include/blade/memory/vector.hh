#ifndef BLADE_MEMORY_VECTOR_HH
#define BLADE_MEMORY_VECTOR_HH

#include "blade/memory/types.hh"

namespace Blade {

class Dimensions : public std::vector<U64> {
 public:
    using std::vector<U64>::vector;

    constexpr const U64 size() const {
        U64 size = 1;
        for (const auto& n : *this) {
            size *= n;
        }
        return size; 
    }
};

template<Device I, typename T, typename Dims> class Vector;

template<typename T, typename Dims>
class VectorImpl : public Dims {
 public:
    VectorImpl()
             : Dims({}),
               container(),
               managed(false) {}
    explicit VectorImpl(const Dims& dims)
             : Dims(dims),
               container(),
               managed(true) {
        this->resize(dims);
    }
    explicit VectorImpl(const std::span<T>& other)
             : Dims({other.size()}),
               container(other),
               managed(false) {}
    explicit VectorImpl(T* ptr, const Dims& dims)
             : Dims(dims),
               container(ptr, size),
               managed(false) {}
    explicit VectorImpl(void* ptr, const Dims& dims)
             : Dims(dims),
               container(static_cast<T*>(ptr), size),
               managed(false) {}

    VectorImpl(const VectorImpl&) = delete;
    // TODO: Check if this works as intended.
    bool operator==(const VectorImpl&) = delete;
    VectorImpl& operator=(const VectorImpl&) = delete;

    virtual ~VectorImpl() {}

    constexpr T* data() const noexcept {
        return container.data();
    }

    constexpr const U64 size() const noexcept {
        return container.size();
    }

    constexpr const U64 size_bytes() const noexcept {
        return container.size_bytes();
    }

    [[nodiscard]] constexpr const bool empty() const noexcept {
        return container.empty();
    }

    constexpr T& operator[](U64 idx) const {
        return container[idx];
    }

    // TODO: Implement iterator.
    constexpr const std::span<T>& span() const {
        return container;
    }

    const Result link(const VectorImpl<T, Dims>& src) {
        if (src.empty()) {
            BL_FATAL("Source can't be empty while linking.");
            return Result::ERROR;
        }

        this->managed = false;
        this->container = src.span();
        static_cast<Dims&>(*this) = src;

        return Result::SUCCESS;
    }

    virtual const Result resize(const Dims& dims) = 0;

    using Dims::Dims;

    constexpr const Dims& dimensions() const noexcept {
        return *this;
    }

 protected:
    std::span<T> container;
    bool managed;
};

}  // namespace Blade

#endif
