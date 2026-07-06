#include <blade/integrator/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <blade/integrator/module.hh>

namespace Jetstream::Blocks {

struct IntegratorImpl : public Block::Impl, public DynamicConfig<Blocks::Integrator> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::Integrator> moduleConfig = std::make_shared<Modules::Integrator>();
};

Result IntegratorImpl::configure() {
    moduleConfig->size = size;
    moduleConfig->rate = rate;
    moduleConfig->axis = axis;
    moduleConfig->blockSize = blockSize;

    return Result::SUCCESS;
}

Result IntegratorImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer", "Input", "Input signal."));
    JST_CHECK(defineInterfaceOutput("buffer", "Output", "Integrated signal."));

    JST_CHECK(defineInterfaceConfig("size",
                                    "Size",
                                    "The number of integrations within each block.",
                                    "int"));
    JST_CHECK(defineInterfaceConfig("rate",
                                    "Rate",
                                    "The number of blocks to integrate together.",
                                    "int"));
    JST_CHECK(defineInterfaceConfig("axis",
                                    "Axis",
                                    "The axis on which to integrate.",
                                    "int"));
    JST_CHECK(defineInterfaceConfig("blockSize",
                                "Block Size",
                                "CUDA threads per block for the integrator kernel.",
                                "int:threads"));

    return Result::SUCCESS;
}

Result IntegratorImpl::create() {
    JST_CHECK(moduleCreate("integrator", moduleConfig, {
        {"buffer", inputs().at("buffer")}
    }));
    JST_CHECK(moduleExposeOutput("buffer", {"integrator", "buffer"}));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(IntegratorImpl);

}  // namespace Jetstream::Blocks
