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
    Pipeline(const U64& numberOfStreams = 2);
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

    template<typename Block>
    void connect(std::shared_ptr<Block>& module,
                 const typename Block::Config& config,
                 const typename Block::Input& input) {
        if (_commited) {
            BL_FATAL("Can't connect new module after Pipeline is commited.");
            BL_CHECK_THROW(Result::ERROR);
        }

        module = std::make_shared<Block>(config, input, _streams[0]);

        if constexpr (std::is_base_of<Bundle, Block>::value) {
            for (auto& mod : module->getModules()) {
                addModule(mod);
            }
        } else {
            addModule(module);
        }
    }

    void addModule(const std::shared_ptr<Module>& module);

    Result compute(const U64& index);
    Result synchronize(const U64& index);
    bool isSynchronized(const U64& index);

    const cudaStream_t& stream(const U64& index = 0) const {
        return _streams[index];
    }

    U64 numberOfStreams() const {
        return _streams.size();
    }

 private:
    bool _commited;
    std::vector<cudaStream_t> _streams;
    std::vector<std::shared_ptr<Module>> modules;

    U64 _computeStepCount;
    U64 _computeStepsPerCycle;
    U64 _computeLifetimeCycles;
    std::vector<U64> _computeStepRatios;

    Result commit();
};

}  // namespace Blade

#endif
