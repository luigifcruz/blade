#ifndef BLADE_MEMORY_COPY_HH
#define BLADE_MEMORY_COPY_HH

#include "blade/memory/types.hh"
#include "blade/memory/vector.hh"

namespace Blade::Memory {

template<Device DstDeviceId, Device SrcDeviceId, typename Type, typename Dims>
static const Result Copy(Vector<DstDeviceId, Type, Dims>& dst,
                         const Vector<SrcDeviceId, Type, Dims>& src,
                         const cudaMemcpyKind& kind,
                         const cudaStream_t& stream = 0) {
    if (dst.size() != src.size()) {
        BL_FATAL("Size mismatch between source and destination ({}, {}).",
                src.size(), dst.size());
    }

    if (dst.shape() != src.shape()) {
        BL_FATAL("Shape mismatch between source ({}) and destination ({}).",
                src.shape(), dst.shape());
    }

    BL_CUDA_CHECK(cudaMemcpyAsync(dst.data(), src.data(), src.size_bytes(),
                kind, stream), [&]{
        BL_FATAL("Can't copy data: {}", err);
        return Result::CUDA_ERROR;
    });

    return Result::SUCCESS;
}

template<typename Type, typename Dims>
static const Result Copy(Vector<Device::CUDA, Type, Dims>& dst,
                         const Vector<Device::CUDA, Type, Dims>& src,
                         const cudaStream_t& stream = 0) {
    return Memory::Copy(dst, src, cudaMemcpyDeviceToDevice, stream);
}

template<typename Type, typename Dims>
static const Result Copy(Vector<Device::CPU, Type, Dims>& dst,
                         const Vector<Device::CPU, Type, Dims>& src,
                         const cudaStream_t& stream = 0) {
    return Memory::Copy(dst, src, cudaMemcpyHostToHost, stream);
}

template<typename Type, typename Dims>
static const Result Copy(Vector<Device::CUDA, Type, Dims>& dst,
                         const Vector<Device::CPU, Type, Dims>& src,
                         const cudaStream_t& stream = 0) {
    return Memory::Copy(dst, src, cudaMemcpyHostToDevice, stream);
}

template<typename Type, typename Dims>
static const Result Copy(Vector<Device::CPU, Type, Dims>& dst,
                         const Vector<Device::CUDA, Type, Dims>& src,
                         const cudaStream_t& stream = 0) {
    return Memory::Copy(dst, src, cudaMemcpyDeviceToHost, stream);
}

template<Device DstDeviceId, Device SrcDeviceId, typename DType, typename SType, typename Dims>
static const Result Copy2D(Vector<DstDeviceId, DType, Dims>& dst,
                           const U64& dstPitch,
                           const U64& dstPad, 
                           const Vector<SrcDeviceId, SType, Dims>& src,
                           const U64& srcPitch,
                           const U64& srcPad,
                           const U64& width,
                           const U64& height,
                           const cudaMemcpyKind& kind,
                           const cudaStream_t& stream = 0) {
    if (width > dstPitch) {
        BL_FATAL("2D copy 'width' is larger than destination's pitch ({}, {}).",
                width, dstPitch);
        return Result::ASSERTION_ERROR;
    }

    if (dst.size_bytes() != (dstPitch * height)) {
        BL_FATAL("Destination's size is not exactly covered by {} rows of {} ({} vs {}).",
                height, dstPitch, dst.size_bytes(), dstPitch * height);
        return Result::ASSERTION_ERROR;
    }

    if (width > srcPitch) {
        BL_FATAL("2D copy 'width' is larger than source's pitch ({}, {}).",
                width, srcPitch);
        return Result::ASSERTION_ERROR;
    }

    if (src.size_bytes() != (srcPitch * height)) {
        BL_FATAL("Source's size is not exactly covered by {} rows of {} ({} vs {}).",
                height, srcPitch, src.size_bytes(), srcPitch * height);
        return Result::ASSERTION_ERROR;
    }

    if (width % sizeof(DType) != 0) {
        BL_FATAL("2D copy 'width' is not a multiple of destination's element size ({}, {}).",
                width, sizeof(DType));
        return Result::ASSERTION_ERROR;
    }

    if (width % sizeof(SType) != 0) {
        BL_FATAL("2D copy 'width' is not a multiple of source's element size ({}, {}).",
                width, sizeof(SType));
        return Result::ASSERTION_ERROR;
    }

    if (srcPad % sizeof(SType) != 0) {
        BL_FATAL("2D copy 'src_pad' is not a multiple of source's element size ({}, {}).",
                srcPad, sizeof(SType));
        return Result::ASSERTION_ERROR;
    }

    if (dstPad % sizeof(DType) != 0) {
        BL_FATAL("2D copy 'dst_pad' is not a multiple of destination's element size ({}, {}).",
                dstPad, sizeof(DType));
        return Result::ASSERTION_ERROR;
    }

    BL_CUDA_CHECK(
        cudaMemcpy2DAsync(
            (uint8_t*)(dst.data()) + dstPad,
            dstPitch,
            (uint8_t*)(src.data()) + srcPad,
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

template<typename DType, typename SType, typename Dims>
static const Result Copy2D(Vector<Device::CUDA, DType, Dims>& dst,
                           const U64& dstPitch,
                           const U64& dstPad, 
                           const Vector<Device::CUDA, SType, Dims>& src,
                           const U64& srcPitch,
                           const U64& srcPad,
                           const U64& width,
                           const U64& height,
                           const cudaStream_t& stream = 0) {
    return Copy2D(dst, dstPitch, dstPad, src, srcPitch, srcPad, 
        width, height, cudaMemcpyDeviceToHost, stream);
}

template<typename DType, typename SType, typename Dims>
static const Result Copy2D(Vector<Device::CUDA, DType, Dims>& dst,
                           const U64& dstPitch,
                           const U64& dstPad, 
                           const Vector<Device::CPU, SType, Dims>& src,
                           const U64& srcPitch,
                           const U64& srcPad,
                           const U64& width,
                           const U64& height,
                           const cudaStream_t& stream = 0) {
    return Copy2D(dst, dstPitch, dstPad, src, srcPitch, srcPad, 
        width, height, cudaMemcpyDeviceToHost, stream);
}

template<typename DType, typename SType, typename Dims>
static const Result Copy2D(Vector<Device::CPU, DType, Dims>& dst,
                           const U64& dstPitch,
                           const U64& dstPad, 
                           const Vector<Device::CUDA, SType, Dims>& src,
                           const U64& srcPitch,
                           const U64& srcPad,
                           const U64& width,
                           const U64& height,
                           const cudaStream_t& stream = 0) {
    return Copy2D(dst, dstPitch, dstPad, src, srcPitch, srcPad, 
        width, height, cudaMemcpyDeviceToHost, stream);
}

}  // namespace Blade::Memory

#endif
