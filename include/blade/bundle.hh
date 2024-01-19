#ifndef BLADE_BUNDLE_HH
#define BLADE_BUNDLE_HH

#include <string>
#include <memory>
#include <vector>

#include "blade/logger.hh"
#include "blade/module.hh"
#include "blade/macros.hh"

namespace Blade {

class BLADE_API Bundle {
 public:
    Bundle(const Stream& stream) : stream(stream) {};

    constexpr std::vector<std::shared_ptr<Module>>& getModules() {
        return modules;
    }

 protected:
    template<typename Block>
    void connect(std::shared_ptr<Block>& module,
                 const typename Block::Config& config,
                 const typename Block::Input& input) {
        module = std::make_unique<Block>(config, input, stream);
        this->modules.push_back(module);
    }

    friend class Pipeline;

 private:
    const Stream stream;
    std::vector<std::shared_ptr<Module>> modules;
};

}  // namespace Blade

#endif
