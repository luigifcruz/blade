#ifndef BLADE_MEMORY_VECTOR_HH
#define BLADE_MEMORY_VECTOR_HH

#include "blade/macros.hh"
#include "blade/memory/types.hh"
#include "blade/memory/shape.hh"

namespace Blade {

template<Device DeviceId,
         typename DataType,
         typename Shape>
struct Vector {
 public:
    Vector()
             : _shape(),
               _data(nullptr),
               _refs(nullptr) {
        BL_TRACE("Empty vector created.");
    }

    explicit Vector(void* ptr, const typename Shape::Type& shape)
             : _shape(shape), 
               _data(static_cast<DataType*>(ptr)),
               _refs(nullptr) {
    }

    explicit Vector(const typename Shape::Type& shape, const bool& unified = false)
             : _shape(shape),
               _data(nullptr),
               _refs(nullptr) {
        BL_TRACE("Vector allocated and created: {}", shape);

        if constexpr (DeviceId == Device::CPU) {
            BL_CUDA_CHECK_THROW(cudaMallocHost(&_data, size_bytes()), [&]{
                BL_FATAL("Failed to allocate pinned host memory: {}", err);
            });
            _refs = new U64(1);
        }

        if constexpr (DeviceId == Device::CUDA) {
            if (unified) {
                BL_CUDA_CHECK_THROW(cudaMallocManaged(&_data, size_bytes()), [&]{
                    BL_FATAL("Failed to allocate managed CUDA memory: {}", err);
                });
            } else {
                BL_CUDA_CHECK_THROW(cudaMalloc(&_data, size_bytes()), [&]{
                    BL_FATAL("Failed to allocate CUDA memory: {}", err);
                });
            }

            BL_CUDA_CHECK_THROW(cudaMallocManaged(&_refs, sizeof(U64)), [&]{
                BL_FATAL("Failed to allocate managed CUDA memory: {}", err);
            });
            *_refs = 1;
        }
    }

    Vector(const Vector& other)
             : _shape(other._shape),
               _data(other._data),
               _refs(other._refs) {
        BL_TRACE("Vector created by copy.");

        increaseRefCount();
    }

    Vector(Vector&& other)
             : _shape(),
               _data(nullptr),
               _refs(nullptr) { 
        BL_TRACE("Vector created by move.");

        std::swap(_data, other._data);
        std::swap(_refs, other._refs);
        std::swap(_shape, other._shape);
    }

    Vector& operator=(const Vector& other) {
        BL_TRACE("Vector copied to existing.");

        decreaseRefCount();
        _data = other._data;
        _refs = other._refs;
        _shape = other._shape;
        increaseRefCount();

        return *this;
    }

    Vector& operator=(Vector&& other) {
        BL_TRACE("Vector moved to existing.");

        decreaseRefCount();
        reset();
        std::swap(_data, other._data);
        std::swap(_refs, other._refs);
        std::swap(_shape, other._shape);

        return *this;
    }

    ~Vector() {
        decreaseRefCount();
    }

    constexpr DataType* data() const noexcept {
        return _data;
    }

    constexpr const U64 refs() const noexcept {
        if (!_refs) {
            return 0;
        }
        return *_refs;
    }

    constexpr const U64 hash() const noexcept {
        return std::hash<void*>{}(_data);
    }

    constexpr const U64 size() const noexcept {
        return _shape.size();
    }

    constexpr const U64 size_bytes() const noexcept {
        return size() * sizeof(DataType);
    }

    constexpr DataType& operator[](const typename Shape::Type& shape) {
        return _data[_shape->shapeToOffset(shape)];
    }

    constexpr const DataType& operator[](const typename Shape::Type& shape) const {
        return this[shape];
    }

    [[nodiscard]] constexpr const bool empty() const noexcept {
        return (_data == nullptr);
    }

    constexpr DataType& operator[](U64 idx) {
        return _data[idx];
    }

    constexpr const DataType& operator[](U64 idx) const {
        return _data[idx];
    }

    constexpr auto begin() {
        return _data;
    }

    constexpr auto end() {
        return _data + size_bytes();
    }

    constexpr const auto begin() const {
        return _data;
    }

    constexpr const auto end() const {
        return _data + size_bytes();
    }

    constexpr const Shape& shape() const {
        return _shape;
    }

    const Result reshape(const Shape& shape) {
        if (shape.size() != _shape.size()) {
            return Result::ERROR;
        }
        _shape = shape;

        return Result::SUCCESS;
    }

 private:
    Shape _shape;
    DataType* _data;
    U64* _refs;

    void decreaseRefCount() {
        if (!_refs) {
            return;
        }

        BL_TRACE("Decreasing reference counter ({}).", *_refs);

        if (--(*_refs) == 0) {
            BL_TRACE("Deleting vector.");

            if constexpr (DeviceId == Device::CPU) {
                if (cudaFreeHost(_data) != cudaSuccess) {
                    BL_FATAL("Failed to deallocate host memory.");
                }
                free(_refs);
            }

            if constexpr (DeviceId == Device::CUDA) {
                if (cudaFree(_data) != cudaSuccess) {
                    BL_FATAL("Failed to deallocate CUDA memory.");
                }

                if (cudaFree(_refs) != cudaSuccess) {
                    BL_FATAL("Failed to deallocate CUDA memory.");
                }
            }

            reset();
        }
    }

    void increaseRefCount() {
        if (!_refs) {
            return;
        }

        BL_TRACE("Increasing reference counter ({}).", *_refs);
        *_refs += 1;
    }

    void reset() {
        _data = nullptr;
        _refs = nullptr;
        _shape = Shape();
    }
};

}  // namespace Blade

#endif
