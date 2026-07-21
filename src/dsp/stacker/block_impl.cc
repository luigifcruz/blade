#include <blade/stacker/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <blade/stacker/module.hh>

namespace Jetstream::Blocks {

struct StackerImpl : public Block::Impl, public DynamicConfig<Blocks::Stacker> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::Stacker> moduleConfig = std::make_shared<Modules::Stacker>();
};

Result StackerImpl::configure() {
    moduleConfig->axis = axis;
    moduleConfig->ratio = ratio;
    moduleConfig->copySizeThreshold = copySizeThreshold;
    moduleConfig->blockSize = blockSize;

    return Result::SUCCESS;
}

Result StackerImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer", "Input", "Polarization complex signal."));
    JST_CHECK(defineInterfaceOutput("buffer", "Output", "Re-oriented polarized complex signal."));

    JST_CHECK(defineInterfaceConfig("axis",
                                    "Axis",
                                    "The axis to stack in the output.",
                                    "uint"));
    JST_CHECK(defineInterfaceConfig("ratio",
                                    "Ratio",
                                    "The ratio of the output to the input in the stacked axis.",
                                    "uint"));
    JST_CHECK(defineInterfaceConfig("copySizeThreshold",
                                    "Copy Threshold",
                                    "The threshold above which CUDA memcpy should be used.",
                                    "uint"));
    JST_CHECK(defineInterfaceConfig("blockSize",
                                "Block Size",
                                "CUDA threads per block for the stacker kernel.",
                                "uint:threads"));

    return Result::SUCCESS;
}

Result StackerImpl::create() {
    JST_CHECK(moduleCreate("stacker", moduleConfig, {
        {"buffer", inputs().at("buffer")}
    }));
    JST_CHECK(moduleExposeOutput("buffer", {"stacker", "buffer"}));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(StackerImpl);

}  // namespace Jetstream::Blocks
