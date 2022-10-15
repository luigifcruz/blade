#include "blade/modules/channelizer.hh"
#include "blade/utils/checker.hh"
#include "blade/pipeline.hh"
#include "blade/memory/base.hh"

using namespace Blade;

template<typename IT, typename OT>
class Test : public Pipeline {
 public:
    typedef typename Modules::Channelizer<IT, OT>::Config Config;

    explicit Test(const Config& config, const ArrayDimensions& arrayDims) {
        BL_CHECK_THROW(this->input.resize(arrayDims));
        this->connect(channelizer, config, {input});
    }

    constexpr const ArrayDimensions& getOutputDims() const {
        return channelizer->getOutputBuffer().dims();
    }

    const Result run(const ArrayTensor<Device::CPU, IT>& input,
                           ArrayTensor<Device::CPU, OT>& output) {
        BL_CHECK(this->copy(this->input, input));
        BL_CHECK(this->compute());
        BL_CHECK(this->copy(output, channelizer->getOutputBuffer()));
        BL_CHECK(this->synchronize());

        return Result::SUCCESS;
    }

 private:
    ArrayTensor<Device::CUDA, IT> input;
    std::shared_ptr<Modules::Channelizer<IT, OT>> channelizer;
};

int main() {
    BL_INFO("Testing advanced channelizer.");

    const ArrayDimensions& arrayDims = {
        .A = 2,
        .F = 4,
        .T = 8,
        .P = 2,
    };

    Test<CF32, CF32> mod({
        .rate = 4,
        .blockSize = 512,
    }, arrayDims);

    ArrayTensor<Device::CPU, CF32> input(arrayDims);
    ArrayTensor<Device::CPU, CF32> output(mod.getOutputDims());

    for (int i = 0; i < 24; i++) {
        if (mod.run(input, output) != Result::SUCCESS) {
            BL_WARN("Fault was encountered. Test is exiting...");
            return 1;
        }
    }

    BL_INFO("Test succeeded.");

    return 0;
}
