#ifndef BLADE_MEMORY_COLLECTION_HH
#define BLADE_MEMORY_COLLECTION_HH

#include <cuda_runtime.h>

#include "blade/memory/types.hh"
#include "blade/memory/vector.hh"

#include "kernels/collection.jit.hh"
#include "blade/utils/jitify2.hh"
using namespace jitify2::reflection;

namespace Blade::Memory {

class BLADE_API Collection {
public:
    template<typename DST_T, typename SRC_T>
    static Result Accumulate(DST_T& dst,
                             const SRC_T& src,
                             const U64& axis,
                             const U64& offset,
                             const cudaStream_t& stream) {
        Collection& instance = getInstance();
        return instance.accumulate(dst, src, axis, offset, stream);
    }

private:
    Collection(const dim3& blockSize = 512)
        : cache(128, *collection_program),
          blockSize(blockSize) {
        BL_TRACE("[MEM::COLLECTION] Initiating.");
    }

    Collection(const Collection&) = delete;
    Collection& operator=(const Collection&) = delete;

    static Collection& getInstance() {
        static Collection instance;
        return instance;
    }

    template<typename T, typename S>
    Result accumulate(Vector<Device::CUDA, T, S>& dst,
                      const Vector<Device::CUDA, T, S>& src,
                      const U64& axis,
                      const U64& offset,
                      const cudaStream_t& stream) {
        // TODO: Fallback to memcpy when number of copies is low.
        cache
            .get_kernel(
                Template("accumulate")
                    .instantiate(
                        TypeInfo<T>::name,
                        S::name,
                        axis,
                        offset
                    )
            )
            ->configure(
                PadGridSize(src.size(), blockSize),
                blockSize,
                0,
                stream
            )
            ->launch(src, dst);

        BL_CUDA_CHECK_KERNEL([&]{
            BL_FATAL("Module failed to execute: {}", err);
            return Result::CUDA_ERROR;
        });

        return Result::SUCCESS;
    }

    static dim3 PadGridSize(const dim3& gridSize, const dim3& blockSize) {
        return dim3((gridSize.x + (blockSize.x - 1)) / blockSize.x,
                    (gridSize.y + (blockSize.y - 1)) / blockSize.y,
                    (gridSize.z + (blockSize.z - 1)) / blockSize.z);
    }

    jitify2::ProgramCache<> cache;
    dim3 blockSize;
};

}  // namespace Blade::Memory

#endif
