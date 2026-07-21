#include <blade/correlator/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <blade/correlator/module.hh>

namespace Jetstream::Blocks {

struct CorrelatorImpl : public Block::Impl, public DynamicConfig<Blocks::Correlator> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::Correlator> moduleConfig = std::make_shared<Modules::Correlator>();
};

Result CorrelatorImpl::configure() {
    moduleConfig->integrationRate = integrationRate;
    moduleConfig->conjugateAntennaIndex = conjugateAntennaIndex;
    moduleConfig->calculationMode = calculationMode;

    return Result::SUCCESS;
}

Result CorrelatorImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer", "Input", "Dual-polarization antenna voltages."));
    JST_CHECK(defineInterfaceOutput("buffer", "Output", "Integrated baseline correlation products."));

    JST_CHECK(defineInterfaceConfig("integrationRate",
                                    "Integration Rate",
                                    "Number of input buffers accumulated into each output visibility buffer.",
                                    "uint:buffers"));
    JST_CHECK(defineInterfaceConfig("conjugateAntennaIndex",
                                    "Conjugate Antenna",
                                    "Select whether the conjugate is applied to antenna A or antenna B.",
                                    "dropdown:0(Antenna A),1(Antenna B)"));
    JST_CHECK(defineInterfaceConfig("calculationMode",
                                    "Calculation Mode",
                                    "Intermediate calculation precision.",
                                    "dropdown:integer(Integer),single_precision_fp(Single Precision FP),double_precision_fp(Double Precision FP)"));

    return Result::SUCCESS;
}

Result CorrelatorImpl::create() {
    JST_CHECK(moduleCreate("correlator", moduleConfig, {
        {"buffer", inputs().at("buffer")}
    }));
    JST_CHECK(moduleExposeOutput("buffer", {"correlator", "buffer"}));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(CorrelatorImpl);

}  // namespace Jetstream::Blocks
