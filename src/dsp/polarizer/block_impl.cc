#include <blade/polarizer/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <blade/polarizer/module.hh>

namespace Jetstream::Blocks {

struct PolarizerImpl : public Block::Impl, public DynamicConfig<Blocks::Polarizer> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::Polarizer> moduleConfig = std::make_shared<Modules::Polarizer>();
};

Result PolarizerImpl::configure() {
    moduleConfig->inputPolarization = inputPolarization;
    moduleConfig->outputPolarization = outputPolarization;
    moduleConfig->blockSize = blockSize;

    return Result::SUCCESS;
}

Result PolarizerImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer", "Input", "Polarization complex signal."));
    JST_CHECK(defineInterfaceOutput("buffer", "Output", "Re-oriented polarized complex signal."));

    JST_CHECK(defineInterfaceConfig("inputPolarization",
                                    "Input Polarization",
                                    "The polarization of the input signal.",
                                    {{"type", "dropdown"}, {"options", Parser::Sequence{
                                        Parser::Map{{"label", "XY"}, {"value", "xy"}},
                                    }}}));
    JST_CHECK(defineInterfaceConfig("outputPolarization",
                                    "Output Polarizations",
                                    "The polarization of the output signal.",
                                    {{"type", "dropdown"}, {"options", Parser::Sequence{
                                        Parser::Map{{"label", "LR"}, {"value", "lr"}},
                                        Parser::Map{{"label", "XY"}, {"value", "xy"}},
                                        Parser::Map{{"label", "X"}, {"value", "x"}},
                                        Parser::Map{{"label", "Y"}, {"value", "y"}},
                                    }}}));
    JST_CHECK(defineInterfaceConfig("blockSize",
                                    "Block Size",
                                    "CUDA threads per block for the polarizer kernel.",
                                    {{"type", "uint"}, {"unit", "threads"}}));

    return Result::SUCCESS;
}

Result PolarizerImpl::create() {
    JST_CHECK(moduleCreate("polarizer", moduleConfig, {
        {"buffer", inputs().at("buffer")}
    }));
    JST_CHECK(moduleExposeOutput("buffer", {"polarizer", "buffer"}));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(PolarizerImpl, {"polarizer"});

}  // namespace Jetstream::Blocks
