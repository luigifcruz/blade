#include <blade/beamformer/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <blade/beamformer/module.hh>

namespace Jetstream::Blocks {

struct BeamformerImpl : public Block::Impl, public DynamicConfig<Blocks::Beamformer> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::Beamformer> moduleConfig = std::make_shared<Modules::Beamformer>();
};

Result BeamformerImpl::configure() {
    moduleConfig->enableIncoherentBeam = enableIncoherentBeam;
    moduleConfig->enableIncoherentBeamSqrt = enableIncoherentBeamSqrt;
    moduleConfig->blockSize = blockSize;

    return Result::SUCCESS;
}

Result BeamformerImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer", "Input", "Complex antenna voltages."));
    JST_CHECK(defineInterfaceInput("phasors", "Phasors", "Beam phasors indexed by beam, antenna, channel, and polarization."));
    JST_CHECK(defineInterfaceOutput("buffer", "Output", "Beamformed complex voltages."));

    JST_CHECK(defineInterfaceConfig("enableIncoherentBeam",
                                    "Incoherent Beam",
                                    "Append an incoherent beam using the first phasor beam as the weighting reference.",
                                    "bool"));
    JST_CHECK(defineInterfaceConfig("enableIncoherentBeamSqrt",
                                    "Incoherent Beam Sqrt",
                                    "Apply a square root to the incoherent beam after power accumulation.",
                                    "bool"));
    JST_CHECK(defineInterfaceConfig("blockSize",
                                    "Block Size",
                                    "CUDA threads per block for the beamformer kernel.",
                                    "int:threads"));

    return Result::SUCCESS;
}

Result BeamformerImpl::create() {
    JST_CHECK(moduleCreate("beamformer", moduleConfig, {
        {"buffer", inputs().at("buffer")},
        {"phasors", inputs().at("phasors")}
    }));
    JST_CHECK(moduleExposeOutput("buffer", {"beamformer", "buffer"}));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(BeamformerImpl);

}  // namespace Jetstream::Blocks
