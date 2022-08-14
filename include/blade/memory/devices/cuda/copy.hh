#ifndef BLADE_MEMORY_CUDA_COPY_HH
#define BLADE_MEMORY_CUDA_COPY_HH

#include "blade/memory/types.hh"
#include "blade/memory/vector.hh"

namespace Blade::Memory {

template<typename T>
static const Result Copy(VectorImpl<T>& dst,
                         const VectorImpl<T>& src,
                         const cudaMemcpyKind& kind,
                         const cudaStream_t& stream = 0) {
    if (dst.size() != src.size()) {
        BL_FATAL("Size mismatch between source and destination ({}, {}).",
                src.size(), dst.size());
    }

    BL_CUDA_CHECK(cudaMemcpyAsync(dst.data(), src.data(), src.size_bytes(),
                kind, stream), [&]{
        BL_FATAL("Can't copy data: {}", err);
        return Result::CUDA_ERROR;
    });

    return Result::SUCCESS;
}

template<typename T>
static const Result Copy(Vector<Device::CUDA, T>& dst,
                         const Vector<Device::CUDA, T>& src,
                         const cudaStream_t& stream = 0) {
    return Memory::Copy(dst, src, cudaMemcpyDeviceToDevice, stream);
}

template<typename T>
static const Result Copy(Vector<Device::CUDA, T>& dst,
                         const Vector<Device::CPU, T>& src,
                         const cudaStream_t& stream = 0) {
    return Memory::Copy(dst, src, cudaMemcpyHostToDevice, stream);
}

template<typename T>
static const Result Copy(Vector<Device::CPU, T>& dst,
                         const Vector<Device::CUDA, T>& src,
                         const cudaStream_t& stream = 0) {
    return Memory::Copy(dst, src, cudaMemcpyDeviceToHost, stream);
}

template<typename T>
static const Result Copy(Vector<Device::CPU, T>& dst,
                         const Vector<Device::CUDA | Device::CPU, T>& src,
                         const cudaStream_t& stream = 0) {
    return Memory::Copy(dst, src, cudaMemcpyHostToHost, stream);
}

template<typename T>
static const Result Copy(Vector<Device::CUDA, T>& dst,
                         const Vector<Device::CUDA | Device::CPU, T>& src,
                         const cudaStream_t& stream = 0) {
    return Memory::Copy(dst, src, cudaMemcpyDeviceToDevice, stream);
}

template<typename T>
static const Result Copy(Vector<Device::CUDA | Device::CPU, T>& dst,
                         const Vector<Device::CUDA, T>& src,
                         const cudaStream_t& stream = 0) {
    return Memory::Copy(dst, src, cudaMemcpyDeviceToDevice, stream);
}

template<typename T>
static const Result Copy(Vector<Device::CUDA | Device::CPU, T>& dst,
                         const Vector<Device::CPU, T>& src,
                         const cudaStream_t& stream = 0) {
    return Memory::Copy(dst, src, cudaMemcpyHostToHost, stream);
}

template<typename T>
static const Result Copy(Vector<Device::CUDA | Device::CPU, T>& dst,
                         const Vector<Device::CUDA | Device::CPU, T>& src,
                         const cudaStream_t& stream = 0) {
    return Memory::Copy(dst, src, cudaMemcpyDeviceToDevice, stream);
}

template<typename DT, typename ST>
static const Result Copy2D(VectorImpl<DT>& dst,
                           const U64& dst_pitch,
                           const U64& dst_pad, 
                           const VectorImpl<ST>& src,
                           const U64& src_pitch,
                           const U64& src_pad,
                           const U64& width,
                           const U64& height,
                           const cudaMemcpyKind& kind,
                           const cudaStream_t& stream = 0) {
    if (width > dst_pitch) {
        BL_FATAL("2D copy 'width' is larger than destination's pitch ({}, {}).",
                width, dst_pitch);
        return Result::ASSERTION_ERROR;
    }

    if (dst.size_bytes() != (dst_pitch * height)) {
        BL_FATAL("Destination's size is not exactly covered by {} rows of {} ({} vs {}).",
                height, dst_pitch, dst.size_bytes(), dst_pitch*height);
        return Result::ASSERTION_ERROR;
    }

    if (width > src_pitch) {
        BL_FATAL("2D copy 'width' is larger than source's pitch ({}, {}).",
                width, src_pitch);
        return Result::ASSERTION_ERROR;
    }

    if (src.size_bytes() != (src_pitch * height)) {
        BL_FATAL("Source's size is not exactly covered by {} rows of {} ({} vs {}).",
                height, src_pitch, src.size_bytes(), src_pitch * height);
        return Result::ASSERTION_ERROR;
    }

    if (width % sizeof(DT) != 0) {
        BL_FATAL("2D copy 'width' is not a multiple of destination's element size ({}, {}).",
                width, sizeof(DT));
        return Result::ASSERTION_ERROR;
    }

    if (width % sizeof(ST) != 0) {
        BL_FATAL("2D copy 'width' is not a multiple of source's element size ({}, {}).",
                width, sizeof(ST));
        return Result::ASSERTION_ERROR;
    }

    if (src_pad % sizeof(ST) != 0) {
        BL_FATAL("2D copy 'src_pad' is not a multiple of source's element size ({}, {}).",
                src_pad, sizeof(ST));
        return Result::ASSERTION_ERROR;
    }

    if (dst_pad % sizeof(DT) != 0) {
        BL_FATAL("2D copy 'dst_pad' is not a multiple of destination's element size ({}, {}).",
                dst_pad, sizeof(DT));
        return Result::ASSERTION_ERROR;
    }

    BL_CUDA_CHECK(
        cudaMemcpy2DAsync(
            dst.data() + dst_pad,
            dst_pitch,
            src.data() + src_pad,
            src_pitch,
            width,
            height,
            kind,
            stream
        ), [&]{
            BL_FATAL("Can't 2D copy data ({}): {}", kind, err);
            return Result::CUDA_ERROR;
        }
    );

    return Result::SUCCESS;
}

template<typename DT, typename ST>
static const Result Copy2D(Vector<Device::CUDA, DT>& dst,
                           const U64& dst_pitch,
                           const U64& dst_pad, 
                           const Vector<Device::CUDA, ST>& src,
                           const U64& src_pitch,
                           const U64& src_pad,
                           const U64& width,
                           const U64& height,
                           const cudaStream_t& stream = 0) {
    return Memory::Copy2D(dst, dst_pitch, dst_pad, src, src_pitch, src_pad, 
        width, height, cudaMemcpyDeviceToHost, stream);
}

template<typename DT, typename ST>
static const Result Copy2D(Vector<Device::CUDA, DT>& dst,
                           const U64& dst_pitch,
                           const U64& dst_pad, 
                           const Vector<Device::CPU, ST>& src,
                           const U64& src_pitch,
                           const U64& src_pad,
                           const U64& width,
                           const U64& height,
                           const cudaStream_t& stream = 0) {
    return Memory::Copy2D(dst, dst_pitch, dst_pad, src, src_pitch, src_pad, 
        width, height, cudaMemcpyDeviceToHost, stream);
}

template<typename DT, typename ST>
static const Result Copy2D(Vector<Device::CPU, DT>& dst,
                           const U64& dst_pitch,
                           const U64& dst_pad, 
                           const Vector<Device::CUDA, ST>& src,
                           const U64& src_pitch,
                           const U64& src_pad,
                           const U64& width,
                           const U64& height,
                           const cudaStream_t& stream = 0) {
    return Memory::Copy2D(dst, dst_pitch, dst_pad, src, src_pitch, src_pad, 
        width, height, cudaMemcpyDeviceToHost, stream);
}

}  // namespace Blade::Memory

#endif
