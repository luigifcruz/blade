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
    Collection(const dim3& blockSize)
         : cache(128, *collection_program),
           blockSize(blockSize) {
    }

    template<typename DST_T, typename SRC_T>
    Result Accumulate(DST_T& dst, const SRC_T& src, const U64& axis, const cudaStream_t& stream) {
        cache
            .get_kernel(Template("accumulate").instantiate())
            ->configure(PadGridSize(src.size(), blockSize),
                        blockSize,
                        0,
                        stream)
            ->launch();

        BL_CUDA_CHECK_KERNEL([&]{
            BL_FATAL("Module failed to execute: {}", err);
            return Result::CUDA_ERROR;
        });
    }

 private:
    jitify2::ProgramCache<> cache;
    dim3 blockSize;
};

}  // namespace Blade::Memory

#endif
