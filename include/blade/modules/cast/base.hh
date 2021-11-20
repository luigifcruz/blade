#ifndef BLADE_MODULES_CAST_GENERIC_H
#define BLADE_MODULES_CAST_GENERIC_H

#include "blade/base.hh"
#include "blade/module.hh"

namespace Blade::Modules {

template<typename IT, typename OT>
class BLADE_API Cast : public Module {
 public:
    struct Config {
        std::size_t blockSize = 512;
    };

    struct Input {
        Memory::DeviceVector<IT>& buf;
    };

    struct Output {
        Memory::DeviceVector<OT> buf;
    };

    explicit Cast(const Config& config, const Input& input);

    constexpr Memory::DeviceVector<IT>& getInput() {
        return this->input.buf;
    }

    constexpr const Memory::DeviceVector<OT>& getOutput() const {
        return this->output.buf;
    }

    constexpr const Config& getConfig() const {
        return this->config;
    }

    Result process(const cudaStream_t& stream = 0) final;

 private:
    const Config config;
    const Input input;
    Output output;
};

}  // namespace Blade::Modules

#endif
