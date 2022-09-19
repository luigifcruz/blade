#include "blade/utils/checker.hh"
#include "blade/modules/beamformer/ata.hh"

using namespace Blade;

template<typename IT, typename OT>
class Test : public Pipeline {
 public:
    typedef typename Modules::Beamformer::ATA<IT, OT>::Config Config;

    explicit Test(const Config& config,
                  const ArrayTensorDimensions& inputDims,
                  const PhasorTensorDimensions& phasorDims) {
        BL_CHECK_THROW(this->input.resize(inputDims));
        BL_CHECK_THROW(this->phasors.resize(phasorDims));
        this->connect(beamformer, config, {input, phasors});
    }

    constexpr const ArrayTensorDimensions& getOutputDims() const {
        return beamformer->getOutputBuffer().dims();
    }

    const Result run(const ArrayTensor<Device::CPU, IT>& input,
                     const PhasorTensor<Device::CPU, IT>& phasors,
                           ArrayTensor<Device::CPU, OT>& output,
                     const bool synchronize = false) {
        BL_CHECK(this->copy(this->input, input));
        BL_CHECK(this->copy(this->phasors, phasors));
        BL_CHECK(this->compute());
        BL_CHECK(this->copy(output, beamformer->getOutputBuffer()));

        if (synchronize) {
            BL_CHECK(this->synchronize());
        }

        return Result::SUCCESS;
    }

 private:
    ArrayTensor<Device::CUDA, IT> input;
    PhasorTensor<Device::CUDA, IT> phasors;

    std::shared_ptr<Modules::Beamformer::ATA<IT, OT>> beamformer;
};

int main() {
    BL_INFO("Testing beamformer with the ATA kernel.");

    const ArrayTensorDimensions& arrayDims = {
        .A = 2,
        .F = 192,
        .T = 512,
        .P = 2,
    };

    const PhasorTensorDimensions& phasorDims = {
        .B = 1,
        .A = 2,
        .F = 192,
        .T = 1,
        .P = 2,
    };

    Test<CF32, CF32> mod({}, arrayDims, phasorDims);

    ArrayTensor<Device::CPU, CF32> input(arrayDims);
    PhasorTensor<Device::CPU, CF32> phasors(phasorDims);
    ArrayTensor<Device::CPU, CF32> output(mod.getOutputDims());

    for (int i = 0; i < 24; i++) {
        if (mod.run(input, phasors, output, true) != Result::SUCCESS) {
            BL_WARN("Fault was encountered. Test is exiting...");
            return 1;
        }
    }

    BL_INFO("Test succeeded.");

    return 0;
}
