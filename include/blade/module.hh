#ifndef BLADE_MODULE_HH
#define BLADE_MODULE_HH

#include <string>

#include "blade/types.hh"
#include "blade/logger.hh"
#include "blade/memory.hh"

#include "blade/utils/jitify2.hh"
using namespace jitify2::reflection;

namespace Blade {

class BLADE_API Module {
 public:
    explicit Module(const std::size_t& blockSize,
                    const jitify2::PreprocessedProgram& kernel);
    virtual ~Module() = default;

    virtual constexpr Result preprocess(const cudaStream_t& stream = 0) {
        return Result::SUCCESS;
    }

    virtual constexpr Result process(const cudaStream_t& stream = 0) {
        return Result::SUCCESS;
    }

 protected:
    jitify2::ProgramCache<> cache;
    std::string kernel;
    dim3 grid, block;

    template<typename T>
    static Result InitInput(T& buffer, std::size_t size) {
        if (buffer.empty()) {
            BL_DEBUG("Input is empty, allocating {} elements", size);
            return buffer.allocate(size);
        }

        if (buffer.size() != size) {
            BL_FATAL("Input size ({}) doesn't match the configuration size ({}).",
                buffer.size(), size);
            return Result::ERROR;
        }

        return Result::SUCCESS;
    }

    template<typename T>
    static Result InitOutput(T& buffer, std::size_t size) {
        if (!buffer.empty()) {
            BL_FATAL("The output buffer should be empty on initialization.");
            return Result::ERROR;
        }

        return buffer.allocate(size);
    }

    template<typename T>
    static const std::string CudaType();

    template<typename T>
    static const std::size_t CudaTypeSize();
};

}  // namespace Blade

#endif
