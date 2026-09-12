#include <blade/detector/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <blade/detector/module.hh>

namespace Jetstream::Blocks {

struct DetectorImpl : public Block::Impl, public DynamicConfig<Blocks::Detector> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::Detector> moduleConfig = std::make_shared<Modules::Detector>();
};

Result DetectorImpl::configure() {
    moduleConfig->integrationRate = integrationRate;
    moduleConfig->numberOfOutputPolarizations = numberOfOutputPolarizations;
    moduleConfig->blockSize = blockSize;

    return Result::SUCCESS;
}

Result DetectorImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer", "Input", "Dual-polarization complex signal."));
    JST_CHECK(defineInterfaceOutput("buffer", "Output", "Integrated detector products."));

    JST_CHECK(defineInterfaceConfig("integrationRate",
                                    "Integration Rate",
                                    "Number of input time samples summed into each detector output sample.",
                                    {{"type", "uint"}, {"unit", "samples"}}));
    JST_CHECK(defineInterfaceConfig("numberOfOutputPolarizations",
                                    "Output Polarizations",
                                    "Select 1 for total power or 4 for XX, YY, Re(XY), Im(XY).",
                                    {{"type", "dropdown"}, {"options", Parser::Sequence{
                                        Parser::Map{{"label", "Total Power"}, {"value", "1"}},
                                        Parser::Map{{"label", "Full Products"}, {"value", "4"}},
                                    }}}));
    JST_CHECK(defineInterfaceConfig("blockSize",
                                    "Block Size",
                                    "CUDA threads per block for the detector kernel.",
                                    {{"type", "uint"}, {"unit", "threads"}}));

    return Result::SUCCESS;
}

Result DetectorImpl::create() {
    JST_CHECK(moduleCreate("detector", moduleConfig, {
        {"buffer", inputs().at("buffer")}
    }));
    JST_CHECK(moduleExposeOutput("buffer", {"detector", "buffer"}));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(DetectorImpl, {"detector"});

}  // namespace Jetstream::Blocks
