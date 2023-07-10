#ifndef BLADE_PIPELINE_HH
#define BLADE_PIPELINE_HH

#include <span>
#include <string>
#include <memory>
#include <vector>

#include "blade/bundle.hh"
#include "blade/logger.hh"
#include "blade/module.hh"

namespace Blade {

class BLADE_API Pipeline {
 public:
    Pipeline(const U64& numberOfComputeSteps = 1);
    virtual ~Pipeline();

    Result synchronize();
    bool isSynchronized();

    constexpr const bool computeComplete() const {
        return currentComputeStep == numberOfComputeSteps;
    }

    constexpr const U64 getComputeNumberOfSteps() const {
        return numberOfComputeSteps;
    }

    constexpr const U64 getCurrentComputeStep() const {
        return currentComputeStep;
    }

    constexpr const U64 getLifetimeComputeCycles() const {
        return lifetimeComputeCycles;
    }

    template<typename Block>
    void connect(std::shared_ptr<Block>& module,
                 const typename Block::Config& config,
                 const typename Block::Input& input) {
        module = std::make_shared<Block>(config, input, stream);

        if constexpr (std::is_base_of<Bundle, Block>::value) {
            for (auto& mod : module->getModules()) {
                modules.push_back(mod);
            }
        } else {
            modules.push_back(module);
        }
    }

 protected:
    Result compute();

    constexpr const cudaStream_t& getCudaStream() const {
        return stream;
    }

    friend class Plan;

 private:
    cudaStream_t stream;
    std::vector<std::shared_ptr<Module>> modules;
    const U64 numberOfComputeSteps;
    U64 currentComputeStep;
    U64 lifetimeComputeCycles;
};

}  // namespace Blade

#endif
