#include "blade/modules/channelizer.hh"
#include "blade/utils/checker.hh"
#include "blade/pipeline.hh"
#include "blade/memory.hh"

using namespace Blade;

template<typename IT, typename OT>
class Test : public Pipeline {
 public:
    explicit Test(const typename Modules::Channelizer<IT, OT>::Config& config) {
        this->connect(channelizer, config, {input});
    }

    constexpr const std::size_t getInputSize() const {
        return channelizer->getBufferSize();
    }

    Result run(const Memory::HostVector<IT>& input,
                     Memory::HostVector<OT>& output) {
        BL_CHECK(this->copy(channelizer->getInput(), input));
        BL_CHECK(this->compute());
        BL_CHECK(this->copy(output, channelizer->getOutput()));
        BL_CHECK(this->synchronize());

        return Result::SUCCESS;
    }

 private:
    Memory::DeviceVector<IT> input;
    std::shared_ptr<Modules::Channelizer<IT, OT>> channelizer;
};

int main() {
    Logger guard{};

    BL_INFO("Testing advanced channelizer.");

    Test<CF32, CF32> mod({
        .dims = {
            .NBEAMS = 1,
            .NANTS  = 20,
            .NCHANS = 96,
            .NTIME  = 35000,
            .NPOLS  = 2,
        },
        .fftSize = 4,
        .blockSize = 512,
    });

    Memory::HostVector<CF32> input(mod.getInputSize());
    Memory::HostVector<CF32> output(mod.getInputSize());

    for (int i = 0; i < 24; i++) {
        if (mod.run(input, output) != Result::SUCCESS) {
            BL_WARN("Fault was encountered. Test is exiting...");
            return 1;
        }
    }

    BL_INFO("Test succeeded.");

    return 0;
}
