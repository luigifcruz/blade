#ifndef BLADE_PIPELINE_HH
#define BLADE_PIPELINE_HH

#include <span>
#include <string>
#include <memory>
#include <vector>

#include "blade/bundle.hh"
#include "blade/logger.hh"
#include "blade/macros.hh"
#include "blade/module.hh"

namespace Blade {

class BLADE_API Pipeline {
 public:
    Pipeline();
    virtual ~Pipeline();

    constexpr const bool computeComplete() const {
        return (_computeStepCount + 1) == _computeStepsPerCycle;
    }

    constexpr const U64& computeCurrentStepCount() const {
        return _computeStepCount;
    }

    constexpr const U64& computeStepsPerCycle() const {
        return _computeStepsPerCycle;
    }

    constexpr const U64& computeLifetimeCycles() const {
        return _computeLifetimeCycles;
    }

    constexpr const cudaStream_t& getCudaStream() const {
        return stream;
    }

    template<typename Block>
    void connect(std::shared_ptr<Block>& module,
                 const typename Block::Config& config,
                 const typename Block::Input& input) {
        if (_commited) {
            BL_FATAL("Can't connect new module after Pipeline is commited.");
            BL_CHECK_THROW(Result::ERROR);
        }

        module = std::make_shared<Block>(config, input, stream);

        if constexpr (std::is_base_of<Bundle, Block>::value) {
            for (auto& mod : module->getModules()) {
                addModule(mod);
            }
        } else {
            addModule(module);
        }
    }

    Result compute();
    Result synchronize();
    bool isSynchronized();

 private:
    cudaStream_t stream;
    std::vector<std::shared_ptr<Module>> modules;
    bool _commited;

    U64 _computeStepCount;
    U64 _computeStepsPerCycle;
    U64 _computeLifetimeCycles;
    std::vector<U64> _computeStepRatios;

    void addModule(const std::shared_ptr<Module>& module);
    Result commit();
};

}  // namespace Blade

#endif
