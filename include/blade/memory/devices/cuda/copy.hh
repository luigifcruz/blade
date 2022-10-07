#ifndef BLADE_MEMORY_CUDA_COPY_HH
#define BLADE_MEMORY_CUDA_COPY_HH

#include "blade/memory/types.hh"
#include "blade/memory/vector.hh"

namespace Blade::Memory {

template<typename T, typename Dims>
static const Result Copy(VectorImpl<T, Dims>& dst,
                         const VectorImpl<T, Dims>& src,
                         const cudaMemcpyKind& kind,
                         const cudaStream_t& stream = 0) {
    if (dst.size() != src.size()) {
        BL_FATAL("Size mismatch between source and destination ({}, {}).",
                src.size(), dst.size());
    }

    if (dst.dims() != src.dims()) {
        BL_FATAL("Dimensions mismatch between source ({}) and destination ({}).",
                src.dims(), dst.dims());
    }

    BL_CUDA_CHECK(cudaMemcpyAsync(dst.data(), src.data(), src.size_bytes(),
                kind, stream), [&]{
        BL_FATAL("Can't copy data: {}", err);
        return Result::CUDA_ERROR;
    });

    return Result::SUCCESS;
}

template<typename T, typename Dims>
static const Result Copy(Vector<Device::CUDA, T, Dims>& dst,
                         const Vector<Device::CUDA, T, Dims>& src,
                         const cudaStream_t& stream = 0) {
    return Memory::Copy(dst, src, cudaMemcpyDeviceToDevice, stream);
}

template<typename T, typename Dims>
static const Result Copy(Vector<Device::CUDA, T, Dims>& dst,
                         const Vector<Device::CPU, T, Dims>& src,
                         const cudaStream_t& stream = 0) {
    return Memory::Copy(dst, src, cudaMemcpyHostToDevice, stream);
}

template<typename T, typename Dims>
static const Result Copy(Vector<Device::CPU, T, Dims>& dst,
                         const Vector<Device::CUDA, T, Dims>& src,
                         const cudaStream_t& stream = 0) {
    return Memory::Copy(dst, src, cudaMemcpyDeviceToHost, stream);
}

template<typename T, typename Dims>
static const Result Copy(Vector<Device::CPU, T, Dims>& dst,
                         const Vector<Device::CUDA | Device::CPU, T, Dims>& src,
                         const cudaStream_t& stream = 0) {
    return Memory::Copy(dst, src, cudaMemcpyHostToHost, stream);
}

template<typename T, typename Dims>
static const Result Copy(Vector<Device::CUDA, T, Dims>& dst,
                         const Vector<Device::CUDA | Device::CPU, T, Dims>& src,
                         const cudaStream_t& stream = 0) {
    return Memory::Copy(dst, src, cudaMemcpyDeviceToDevice, stream);
}

template<typename T, typename Dims>
static const Result Copy(Vector<Device::CUDA | Device::CPU, T, Dims>& dst,
                         const Vector<Device::CUDA, T, Dims>& src,
                         const cudaStream_t& stream = 0) {
    return Memory::Copy(dst, src, cudaMemcpyDeviceToDevice, stream);
}

template<typename T, typename Dims>
static const Result Copy(Vector<Device::CUDA | Device::CPU, T, Dims>& dst,
                         const Vector<Device::CPU, T, Dims>& src,
                         const cudaStream_t& stream = 0) {
    return Memory::Copy(dst, src, cudaMemcpyHostToHost, stream);
}

template<typename T, typename Dims>
static const Result Copy(Vector<Device::CUDA | Device::CPU, T, Dims>& dst,
                         const Vector<Device::CUDA | Device::CPU, T, Dims>& src,
                         const cudaStream_t& stream = 0) {
    return Memory::Copy(dst, src, cudaMemcpyDeviceToDevice, stream);
}

template<typename DT, typename ST, typename Dims>
static const Result Copy2D(VectorImpl<DT, Dims>& dst,
                           const U64& dstPitch,
                           const U64& dstOffset, 
                           const VectorImpl<ST, Dims>& src,
                           const U64& srcPitch,
                           const U64& srcOffset,
                           const U64& width,
                           const U64& height,
                           const cudaMemcpyKind& kind,
                           const cudaStream_t& stream = 0) {
    if (width > dstPitch) {
        BL_FATAL("2D copy 'width' is larger than destination's pitch ({}, {}).",
                width, dstPitch);
        return Result::ASSERTION_ERROR;
    }

    if (dst.size_bytes() < (dstPitch * height)) {
        BL_FATAL("Destination's size is surpassed by {} rows of {} ({} vs {}).",
                height, dstPitch, dst.size_bytes(), dstPitch * height);
        return Result::ASSERTION_ERROR;
    }

    if (width > srcPitch) {
        BL_FATAL("2D copy 'width' is larger than source's pitch ({}, {}).",
                width, srcPitch);
        return Result::ASSERTION_ERROR;
    }

    if (src.size_bytes() < (srcPitch * height)) {
        BL_FATAL("Source's size is surpassed by {} rows of {} ({} vs {}).",
                height, srcPitch, src.size_bytes(), srcPitch * height);
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

    if (srcOffset % sizeof(ST) != 0) {
        BL_FATAL("2D copy source offset (bytes) is not a multiple of source's element size ({}, {}).",
                srcOffset, sizeof(ST));
        return Result::ASSERTION_ERROR;
    }

    if (dstOffset % sizeof(DT) != 0) {
        BL_FATAL("2D copy destination offset (bytes) is not a multiple of destination's element size ({}, {}).",
                dstOffset, sizeof(DT));
        return Result::ASSERTION_ERROR;
    }

    BL_CUDA_CHECK(
        cudaMemcpy2DAsync(
            (uint8_t*)(dst.data()) + dstOffset,
            dstPitch,
            (uint8_t*)(src.data()) + srcOffset,
            srcPitch,
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

template<typename DT, typename ST, typename Dims>
static const Result Copy2D(Vector<Device::CUDA, DT, Dims>& dst,
                           const U64& dstPitch,
                           const U64& dstOffset, 
                           const Vector<Device::CUDA, ST, Dims>& src,
                           const U64& srcPitch,
                           const U64& srcOffset,
                           const U64& width,
                           const U64& height,
                           const cudaStream_t& stream = 0) {
    return Memory::Copy2D(dst, dstPitch, dstOffset, src, srcPitch, srcOffset, 
        width, height, cudaMemcpyDeviceToHost, stream);
}

template<typename DT, typename ST, typename Dims>
static const Result Copy2D(Vector<Device::CUDA, DT, Dims>& dst,
                           const U64& dstPitch,
                           const U64& dstOffset, 
                           const Vector<Device::CPU, ST, Dims>& src,
                           const U64& srcPitch,
                           const U64& srcOffset,
                           const U64& width,
                           const U64& height,
                           const cudaStream_t& stream = 0) {
    return Memory::Copy2D(dst, dstPitch, dstOffset, src, srcPitch, srcOffset, 
        width, height, cudaMemcpyDeviceToHost, stream);
}

template<typename DT, typename ST, typename Dims>
static const Result Copy2D(Vector<Device::CPU, DT, Dims>& dst,
                           const U64& dstPitch,
                           const U64& dstOffset, 
                           const Vector<Device::CUDA, ST, Dims>& src,
                           const U64& srcPitch,
                           const U64& srcOffset,
                           const U64& width,
                           const U64& height,
                           const cudaStream_t& stream = 0) {
    return Memory::Copy2D(dst, dstPitch, dstOffset, src, srcPitch, srcOffset, 
        width, height, cudaMemcpyDeviceToHost, stream);
}

}  // namespace Blade::Memory

#endif
