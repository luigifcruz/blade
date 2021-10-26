#include "blade/channelizer/test.hh"
#include "blade/channelizer/base.hh"
#include "blade/checker/base.hh"
#include "blade/manager.hh"
#include "blade/pipeline.hh"

using namespace Blade;

class Module : public Pipeline {
public:
    Module(const Channelizer::Config& config) : config(config) {
        if (this->commit() != Result::SUCCESS) {
            throw Result::ERROR;
        }
    }

protected:
    Result underlyingInit() final {
        BL_INFO("Initializing kernels.");

        channelizer = std::make_unique<Channelizer>(config);
        test = std::make_unique<Channelizer::Test>(config);
        checker = Factory<Checker>({});

        return Result::SUCCESS;
    }

    Result underlyingAllocate() final {
        BL_INFO("Allocating resources.");
        BL_CHECK(allocateBuffer(input, channelizer->getBufferSize()));
        BL_CHECK(allocateBuffer(output, channelizer->getBufferSize()));
        BL_CHECK(allocateBuffer(result, channelizer->getBufferSize()));

        BL_INFO("Generating test data with Python.");
        BL_CHECK(test->process());

        BL_INFO("Copying test data to the device.");
        BL_CHECK(copyBuffer(input, test->getInputData(), CopyKind::H2D));
        BL_CHECK(copyBuffer(result, test->getOutputData(), CopyKind::H2D));

        return Result::SUCCESS;
    }

    Result underlyingReport(Resources& res) final {
        BL_INFO("Reporting resources.");

        res.transfer.h2d += input.size_bytes();
        res.transfer.d2h += output.size_bytes();

        return Result::SUCCESS;
    }

    Result underlyingProcess(cudaStream_t& cudaStream) final {
        BL_CHECK(channelizer->run(input, output, cudaStream));

        return Result::SUCCESS;
    }

    Result underlyingPostprocess() final {
        std::size_t errors = 0;
        if ((errors = checker->run(output, result)) != 0) {
            BL_FATAL("Module produced {} errors.", errors);
            return Result::ERROR;
        }

        return Result::SUCCESS;
    }

private:
    const Channelizer::Config& config;

    std::unique_ptr<Channelizer> channelizer;
    std::unique_ptr<Channelizer::Test> test;
    std::unique_ptr<Checker> checker;

    std::span<std::complex<float>> input;
    std::span<std::complex<float>> output;
    std::span<std::complex<float>> result;
};

int main() {
    Logger guard{};
    Manager manager{};

    BL_INFO("Testing advanced channelizer.");

    Module mod({
        {
            .NBEAMS = 1,
            .NANTS  = 20,
            .NCHANS = 96,
            .NTIME  = 35000,
            .NPOLS  = 2,
        }, {
            .fftSize = 4,
            .blockSize = 1024,
        },
    });

    manager.save(mod).report();

    for (int i = 0; i < 150; i++) {
        if (mod.process(true) != Result::SUCCESS) {
            BL_WARN("Fault was encountered. Test is exiting...");
            return 1;
        }
    }

    BL_INFO("Test succeeded.");

    return 0;
}
