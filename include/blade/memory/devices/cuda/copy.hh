#ifndef BLADE_MEMORY_CUDA_COPY_HH
#define BLADE_MEMORY_CUDA_COPY_HH

#include "blade/memory/types.hh"
#include "blade/memory/vector.hh"

namespace Blade::Memory {

template<typename T>
static Result Copy(VectorImpl<T>& dst,
                   const VectorImpl<T>& src,
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
        return Result::CUDA_ERROR;
    });

    return Result::SUCCESS;
}

template<typename T>
static Result Copy(Vector<Device::CUDA, T>& dst,
                   const Vector<Device::CUDA, T>& src,
                   const cudaStream_t& stream = 0) {
    return Memory::Copy(dst, src, cudaMemcpyDeviceToDevice, stream);
}

template<typename T>
static Result Copy(Vector<Device::CUDA, T>& dst,
                   const Vector<Device::CPU, T>& src,
                   const cudaStream_t& stream = 0) {
    return Memory::Copy(dst, src, cudaMemcpyHostToDevice, stream);
}

template<typename T>
static Result Copy(Vector<Device::CPU, T>& dst,
                   const Vector<Device::CUDA, T>& src,
                   const cudaStream_t& stream = 0) {
    return Memory::Copy(dst, src, cudaMemcpyDeviceToHost, stream);
}

template<typename DT, typename ST>
static Result Copy2D(VectorImpl<DT>& dst,
                   const U64 dpitch,
                   const VectorImpl<ST>& src,
                   const U64 spitch,
                   const U64 width,
                   const U64 height,
                   const cudaMemcpyKind& kind,
                   const cudaStream_t& stream = 0) {
    auto failure = false;
    if (width > dpitch) {
        BL_FATAL("2D copy 'width' is larger than destination's pitch ({}, {}).",
                width, dpitch);
        failure = true;
    }
    if (dst.size_bytes() != dpitch*height) {
        BL_FATAL("Destination's size is not exactly covered by {} rows of {} ({} vs {}).",
                height, dpitch, dst.size_bytes(), dpitch*height);
        failure = true;
    }
    if (width > spitch) {
        BL_FATAL("2D copy 'width' is larger than source's pitch ({}, {}).",
                width, spitch);
        failure = true;
    }
    if (src.size_bytes() != spitch*height) {
        BL_FATAL("Source's size is not exactly covered by {} rows of {} ({} vs {}).",
                height, spitch, src.size_bytes(), spitch*height);
        failure = true;
    }
    if (width % sizeof(DT) != 0) {
        BL_FATAL("2D copy 'width' is not a multiple of destination's element size ({}, {}).",
                width, sizeof(DT));
        failure = true;
    }
    if (width % sizeof(ST) != 0) {
        BL_FATAL("2D copy 'width' is not a multiple of source's element size ({}, {}).",
                width, sizeof(ST));
        failure = true;
    }
    if (failure) {
        return Result::ASSERTION_ERROR;
    }

    BL_CUDA_CHECK(
        cudaMemcpy2DAsync(
            dst.data(),
            dpitch,
            src.data(),
            spitch,
            width,
            height,
            kind, stream),
        [&]{
            BL_FATAL("Can't 2D copy data ({}): {}", kind, err);
            return Result::CUDA_ERROR;
        }
    );

    return Result::SUCCESS;
}

template<typename DT, typename ST>
static Result Copy2D(Vector<Device::CUDA, DT>& dst,
                   const U64 dpitch,
                   const Vector<Device::CUDA, ST>& src,
                   const U64 spitch,
                   const U64 width,
                   const U64 height,
                   const cudaStream_t& stream = 0) {
    return Memory::Copy2D(dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToDevice, stream);
}

template<typename DT, typename ST>
static Result Copy2D(Vector<Device::CUDA, DT>& dst,
                   const U64 dpitch,
                   const Vector<Device::CPU, ST>& src,
                   const U64 spitch,
                   const U64 width,
                   const U64 height,
                   const cudaStream_t& stream = 0) {
    return Memory::Copy2D(dst, dpitch, src, spitch, width, height, cudaMemcpyHostToDevice, stream);
}

template<typename DT, typename ST>
static Result Copy2D(Vector<Device::CPU, DT>& dst,
                   const U64 dpitch,
                   const Vector<Device::CUDA, ST>& src,
                   const U64 spitch,
                   const U64 width,
                   const U64 height,
                   const cudaStream_t& stream = 0) {
    return Memory::Copy2D(dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToHost, stream);
}

}  // namespace Blade::Memory

#endif
