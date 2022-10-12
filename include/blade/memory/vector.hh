#ifndef BLADE_MEMORY_VECTOR_HH
#define BLADE_MEMORY_VECTOR_HH

#include "blade/memory/types.hh"

namespace Blade {

template <typename T>
concept IsDimensions = 
requires(T t) {
    { t.size() } -> std::same_as<U64>;
    { t == t } -> std::same_as<BOOL>;
};

template<typename T, typename Dims>
requires IsDimensions<Dims>
class VectorImpl {
 public:
    VectorImpl()
             : dimensions(),
               container(),
               managed(true) {}
    explicit VectorImpl(const std::span<T>& other, const Dims& dims)
             : dimensions(dims),
               container(other),
               managed(false) {
        BL_CHECK_THROW(dims.size() == other.size() ? Result::SUCCESS : Result::ERROR);
    }
    explicit VectorImpl(T* ptr, const Dims& dims)
             : dimensions(dims),
               container(ptr, dims.size()),
               managed(false) {}
    explicit VectorImpl(void* ptr, const Dims& dims)
             : dimensions(dims),
               container(static_cast<T*>(ptr), dims.size()),
               managed(false) {}

    VectorImpl(const VectorImpl&) = delete;
    VectorImpl(const VectorImpl&&) = delete;
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

    constexpr auto begin() {
        return container.begin();
    }

    constexpr auto end() {
        return container.end();
    }

    constexpr const auto begin() const {
        return container.begin();
    }

    constexpr const auto end() const {
        return container.end();
    }

    const Result link(const VectorImpl<T, Dims>& src) {
        if (src.empty()) {
            BL_FATAL("Source can't be empty while linking.");
            return Result::ERROR;
        }

        this->managed = false;
        this->container = src.span();
        this->dimensions = src.dims();

        return Result::SUCCESS;
    }

    virtual const Result resize(const Dims& dims) = 0;

    constexpr const Dims& dims() const {
        return dimensions;
    }

 protected:
    Dims dimensions;
    std::span<T> container;
    bool managed;

    explicit VectorImpl(const Dims& dims)
             : dimensions(dims),
               container(),
               managed(true) {}

    constexpr const std::span<T>& span() const {
        return container;
    }
};

}  // namespace Blade

#endif
