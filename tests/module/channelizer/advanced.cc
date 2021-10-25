#include "blade/channelizer/test.hh"
#include "blade/channelizer/base.hh"
#include "blade/checker/base.hh"
#include "blade/manager.hh"
#include "blade/pipeline.hh"

using namespace Blade;

class Module : public Pipeline {
public:
    Module(Channelizer::Generic &channelizer,
           Channelizer::Test::Generic &test) :
        channelizer(channelizer),
        test(test),
        checker({}) {
        if (this->commit() != Result::SUCCESS) {
            throw Result::ERROR;
        }
    };

    ~Module() {
        Free(input);
        Free(output);
        Free(result);
        Free(counter);
    }

    Resources getResources() {
        Resources res;

        // Report device memory.
        res.memory.device += input.size_bytes();
        res.memory.device += output.size_bytes();
        res.memory.device += result.size_bytes();
        res.memory.device += counter.size_bytes();

        // Report host memory.
        res.memory.host += input.size_bytes();
        res.memory.host += output.size_bytes();
        res.memory.host += result.size_bytes();
        res.memory.host += counter.size_bytes();

        // Report transfers.
        res.transfer.h2d += input.size_bytes();
        res.transfer.d2h += output.size_bytes();

        return res;
    }

    unsigned long long int getCounter() const {
        return counter[0];
    }

    void resetCounter() const {
        counter[0] = 0;
    }

protected:
    Result underlyingAllocate() {
        BL_CHECK(Allocate(channelizer.getBufferSize(), input));
        BL_CHECK(Allocate(channelizer.getBufferSize(), output));
        BL_CHECK(Allocate(channelizer.getBufferSize(), result));
        BL_CHECK(Allocate(2, counter, true));

        BL_INFO("Generating test data with Python...");
        BL_CHECK(test.process());

        BL_INFO("Copying test data to the device...");
        BL_CHECK(Transfer(input, test.getInputData(), CopyKind::H2D));
        BL_CHECK(Transfer(result, test.getOutputData(), CopyKind::H2D));

        return Result::SUCCESS;
    }

    Result underlyingProcess(cudaStream_t &cudaStream) {
        BL_CHECK(channelizer.run(input, output, cudaStream));
        BL_CHECK(checker.run(output, result, counter, cudaStream));

        return Result::SUCCESS;
    }

private:
    Channelizer::Generic &channelizer;
    Channelizer::Test::Generic &test;
    Checker::Generic checker;

    std::span<std::complex<float>> input;
    std::span<std::complex<float>> output;
    std::span<std::complex<float>> result;
    std::span<std::size_t> counter;
};

int main() {
    Logger guard{};

    BL_INFO("Testing advanced channelizer.");

    Channelizer::Generic channelizer({
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

    Channelizer::Test::Generic test(channelizer.getConfig());

    Module mod(channelizer, test);
    mod.resetCounter();

    for (int i = 0; i < 150; i++) {
        if (mod.process() != Result::SUCCESS) {
            BL_WARN("Fault was encountered. Test is exiting...");
            return 1;
        }

        if (mod.getCounter() != 0) {
            BL_FATAL("Beamformer produced {} errors.", mod.getCounter());
            return 1;
        }
    }

    BL_INFO("Test succeeded.");

    return 0;
}
