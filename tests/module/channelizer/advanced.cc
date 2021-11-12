#include "blade/modules/channelizer/test.hh"
#include "blade/modules/channelizer/base.hh"
#include "blade/utils/checker.hh"
#include "blade/manager.hh"
#include "blade/pipeline.hh"

using namespace Blade;

class Module : public Pipeline {
 public:
    explicit Module(const Modules::Channelizer::Config& config) :
        Pipeline(false, true), config(config) {
        if (this->setup() != Result::SUCCESS) {
            throw Result::ERROR;
        }
    }

    Result run() {
        return this->loop();
    }

 protected:
    Result setupModules() final {
        BL_INFO("Initializing kernels.");

        channelizer = std::make_unique<Modules::Channelizer>(config);

        return Result::SUCCESS;
    }

    Result setupMemory() final {
        BL_INFO("Allocating resources.");
        BL_CHECK(allocateBuffer(input, channelizer->getBufferSize()));
        BL_CHECK(allocateBuffer(output, channelizer->getBufferSize(), true));

        return Result::SUCCESS;
    }

    Result setupTest() final {
        test = std::make_unique<Modules::Channelizer::Test>(config);

        BL_CHECK(test->process());
        BL_CHECK(copyBuffer(input, test->getInputData(), CopyKind::H2D));

        return Result::SUCCESS;
    }

    Result loopProcess(cudaStream_t& cudaStream) final {
        BL_CHECK(channelizer->run(input, output, cudaStream));

        return Result::SUCCESS;
    }

    Result loopTest() final {
        std::size_t errors = 0;
        if ((errors = Checker::run(output, test->getOutputData())) != 0) {
            BL_FATAL("Module produced {} errors.", errors);
            return Result::ERROR;
        }

        return Result::SUCCESS;
    }

 private:
    const Modules::Channelizer::Config& config;

    std::unique_ptr<Modules::Channelizer> channelizer;
    std::unique_ptr<Modules::Channelizer::Test> test;

    std::span<CF32> input;
    std::span<CF32> output;
};

int main() {
    Logger guard{};
    Manager manager{};

    BL_INFO("Testing advanced channelizer.");

    Module mod({
        .dims = {
            .NBEAMS = 1,
            .NANTS  = 20,
            .NCHANS = 96,
            .NTIME  = 35000,
            .NPOLS  = 2,
        },
        .fftSize = 4,
        .blockSize = 1024,
    });

    manager.save(mod.getResources()).report();

    for (int i = 0; i < 24; i++) {
        if (mod.run() != Result::SUCCESS) {
            BL_WARN("Fault was encountered. Test is exiting...");
            return 1;
        }
    }

    BL_INFO("Test succeeded.");

    return 0;
}
