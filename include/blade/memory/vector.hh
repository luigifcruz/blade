#ifndef BLADE_MEMORY_VECTOR_HH
#define BLADE_MEMORY_VECTOR_HH

#include <span>

#include "blade/macros.hh"
#include "blade/memory/types.hh"
#include "blade/memory/shape.hh"

namespace Blade {

template<Device DeviceId, typename DataType, typename ShapeClass>
struct Vector : public ShapeClass {
 public:
    Vector()
             : ShapeClass(),
               managed(false),
               _data(nullptr) {
        BL_TRACE("Empty vector created.");
    }

    Vector(Vector& vector)
             : ShapeClass(vector.shape()),
               managed(false),
               _data(vector.data()) {
        BL_TRACE("Vector copied.");
    }

    Vector(const Vector& vector)
             : ShapeClass(vector.shape()), 
               managed(false), 
               _data(vector.data()) {
        BL_TRACE("Vector const copied.");
    }

    Vector(DataType* ptr,
           const typename ShapeClass::Type& shape)
             : ShapeClass(shape), 
               managed(false),
               _data(ptr) {
        BL_TRACE("Vector created.");
    }

    Vector(void* ptr,
           const typename ShapeClass::Type& shape)
             : ShapeClass(shape), 
               managed(false),
               _data(static_cast<DataType*>(ptr)) {
        BL_TRACE("Vector created.");
    }

    Vector(const ShapeClass& shape, const bool& unified = false)
             : ShapeClass(shape.shape()),
               managed(true) {
        BL_TRACE("Vector allocated and created.");
        BL_CHECK_THROW(this->allocate(unified));
    }

    Vector(const typename ShapeClass::Type& shape, const bool& unified = false)
             : ShapeClass(shape),
               managed(true) {
        BL_TRACE("Vector allocated and created.");
        BL_CHECK_THROW(this->allocate(unified));
    }

    Vector& operator=(Vector&& other) {
        if (!empty()) {
            BL_FATAL("Can't move contents to a managed vector.");
            BL_CHECK_THROW(Result::ERROR);
        }

        std::swap(this->_shape, other._shape);
        std::swap(_data, other._data);
        std::swap(managed, other.managed);

        return *this;
    }

    Vector& operator=(Vector&) = delete;
    bool operator==(const Vector&) = delete;

    ~Vector() {
        BL_TRACE("Deleting vector.")

        if (!managed || !_data) {
            BL_TRACE("Vector isn't managed.");
            return;
        }

        if constexpr (DeviceId == Device::CPU) {
            if (cudaFreeHost(_data) != cudaSuccess) {
                BL_FATAL("Failed to deallocate host memory.");
            }
        }

        if constexpr (DeviceId == Device::CUDA) {
            if (cudaFree(_data) != cudaSuccess) {
                BL_FATAL("Failed to deallocate CUDA memory.");
            }
        }
    }

    constexpr DataType* data() const noexcept {
        return _data;
    }

    constexpr const U64 size_bytes() const noexcept {
        return this->size() * sizeof(DataType);
    }

    constexpr DataType& operator[](const typename ShapeClass::Type& shape) {
        return _data[this->shapeToOffset(shape)];
    }

    constexpr const DataType& operator[](const typename ShapeClass::Type shape) const {
        return this[shape];
    }

    [[nodiscard]] constexpr const bool empty() const noexcept {
        return (managed == false) && (_data == nullptr);
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

 private:
    bool managed;
    DataType* _data;

    const Result allocate(const bool& unified = false) {
        if constexpr (DeviceId == Device::CPU) {
            BL_CUDA_CHECK(cudaMallocHost(&_data, size_bytes()), [&]{
                BL_FATAL("Failed to allocate pinned host memory: {}", err);
            });
        }

        if constexpr (DeviceId == Device::CUDA) {
            if (unified) {
                BL_CUDA_CHECK(cudaMallocManaged(&_data, size_bytes()), [&]{
                    BL_FATAL("Failed to allocate managed CUDA memory: {}", err);
                });
            } else {
                BL_CUDA_CHECK(cudaMalloc(&_data, size_bytes()), [&]{
                    BL_FATAL("Failed to allocate CUDA memory: {}", err);
                });
            }
        }

        return Result::SUCCESS;
    }
};

}  // namespace Blade

#endif
