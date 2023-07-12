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
        return _computeCurrentStepNumber == _computeTotalNumberOfSteps;
    }

    constexpr const U64& computeTotalNumberOfSteps() const {
        return _computeTotalNumberOfSteps;
    }

    constexpr const U64& computeCurrentStepNumber() const {
        return _computeCurrentStepNumber;
    }

    constexpr const U64& computeNumberOfLifetimeCycles() const {
        return _computeNumberOfLifetimeCycles;
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
    U64 _computeTotalNumberOfSteps;
    U64 _computeCurrentStepNumber;
    U64 _computeNumberOfLifetimeCycles;

    void addModule(const std::shared_ptr<Module>& module);
    Result commit();
};

}  // namespace Blade

#endif
