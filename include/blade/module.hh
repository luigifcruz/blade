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
    static const std::string cudaType();

    template<typename T>
    static const std::size_t cudaTypeSize();
};

}  // namespace Blade

#endif
