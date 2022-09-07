#include "blade/modules/channelizer.hh"
#include "blade/utils/checker.hh"
#include "blade/pipeline.hh"
#include "blade/memory/base.hh"

using namespace Blade;

template<typename IT, typename OT>
class Test : public Pipeline {
 public:
    explicit Test(const typename Modules::Channelizer<IT, OT>::Config& config) {
        this->connect(channelizer, config, {input});
    }

    constexpr const U64 getInputDims() const {
        return channelizer->getInput();
    }

    const Result run(const ArrayTensor<Device::CPU, IT>& input,
                           ArrayTensor<Device::CPU, OT>& output) {
        BL_CHECK(this->copy(channelizer->getInput(), input));
        BL_CHECK(this->compute());
        BL_CHECK(this->copy(output, channelizer->getOutput()));
        BL_CHECK(this->synchronize());

        return Result::SUCCESS;
    }

 private:
    ArrayTensor<Device::CUDA, IT> input;
    std::shared_ptr<Modules::Channelizer<IT, OT>> channelizer;
};

int main() {
    BL_INFO("Testing advanced channelizer.");

    Test<CF32, CF32> mod({
        .numberOfBeams = 1,
        .numberOfAntennas = 20,
        .numberOfFrequencyChannels = 96,
        .numberOfTimeSamples = 35000,
        .numberOfPolarizations = 2,
        .rate = 4,
        .blockSize = 512,
    });

    ArrayTensor<Device::CPU, CF32> input(mod.getInputSize());
    ArrayTensor<Device::CPU, CF32> output(mod.getInputSize());

    for (int i = 0; i < 24; i++) {
        if (mod.run(input, output) != Result::SUCCESS) {
            BL_WARN("Fault was encountered. Test is exiting...");
            return 1;
        }
    }

    BL_INFO("Test succeeded.");

    return 0;
}
