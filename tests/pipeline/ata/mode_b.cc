#include <memory>
#include <chrono>

#include "blade/cast/base.hh"
#include "blade/checker/base.hh"
#include "blade/channelizer/base.hh"
#include "blade/beamformer/ata.hh"
#include "blade/pipeline.hh"
#include "blade/manager.hh"

using namespace Blade;
using namespace std::chrono;

class ModeB : public Pipeline {
 public:
    struct Config {
        ArrayDims inputDims;
        std::size_t channelizerRate = 4;
        std::size_t beamformerBeams = 16;
        std::size_t castBlockSize = 512;
        std::size_t channelizerBlockSize = 512;
        std::size_t beamformerBlockSize = 350;
    };

    explicit ModeB(const Config& configuration) : config(configuration) {
        if (this->commit() != Result::SUCCESS) {
            throw Result::ERROR;
        }
    }

    std::size_t getInputSize() const {
        return channelizer->getBufferSize();
    }

    std::size_t getOutputSize() const {
        return beamformer->getOutputSize();
    }

    Result upload(const std::span<CI8>& input) {
        BL_CHECK(copyBuffer(bufferA, input, CopyKind::H2D));

        return Result::SUCCESS;
    }

    Result download(std::span<CF16> output) {
        BL_CHECK(copyBuffer(output, bufferE, CopyKind::D2H));

        return Result::SUCCESS;
    }

 protected:
    Result underlyingInit() final {
        BL_INFO("Initializing kernels.");

        cast = Factory<Cast>({
            .blockSize = config.castBlockSize,
        });

        channelizer = Factory<Channelizer>({
            .dims = config.inputDims,
            .fftSize = config.channelizerRate,
            .blockSize = config.channelizerBlockSize,
        });

        // TODO: This is a hack.
        auto dims = channelizer->getOutputDims();
        dims.NBEAMS *= 16;
        //

        beamformer = Factory<Beamformer::ATA>({
            .dims = dims,
            .blockSize = config.beamformerBlockSize,
        });

        return Result::SUCCESS;
    }

    Result underlyingAllocate() final {
        BL_INFO("Allocating resources.");

        BL_CHECK(allocateBuffer(phasors, beamformer->getPhasorsSize()));
        BL_CHECK(allocateBuffer(bufferA, channelizer->getBufferSize()));
        BL_CHECK(allocateBuffer(bufferB, channelizer->getBufferSize()));
        BL_CHECK(allocateBuffer(bufferC, channelizer->getBufferSize()));
        BL_CHECK(allocateBuffer(bufferD, beamformer->getOutputSize()));
        BL_CHECK(allocateBuffer(bufferE, beamformer->getOutputSize()));

        return Result::SUCCESS;
    }

    Result underlyingReport(Resources& res) final {
        BL_INFO("Reporting resources.");

        res.transfer.h2d += bufferA.size_bytes();
        res.transfer.d2h += bufferE.size_bytes();

        return Result::SUCCESS;
    }

    Result underlyingProcess(cudaStream_t& cudaStream) final {
        BL_CHECK(cast->run(bufferA, bufferB, cudaStream));
        BL_CHECK(channelizer->run(bufferB, bufferC, cudaStream));
        BL_CHECK(beamformer->run(bufferC, phasors, bufferD, cudaStream));
        BL_CHECK(cast->run(bufferD, bufferE, cudaStream));

        return Result::SUCCESS;
    }

 private:
    const Config& config;

    std::span<CF32> phasors;
    std::span<CI8> bufferA;
    std::span<CF32> bufferB;
    std::span<CF32> bufferC;
    std::span<CF32> bufferD;
    std::span<CF16> bufferE;

    std::unique_ptr<Cast> cast;
    std::unique_ptr<Beamformer::ATA> beamformer;
    std::unique_ptr<Channelizer> channelizer;
};

int main() {
    Logger guard{};
    Manager manager{};

    BL_INFO("Testing ATA beamformer pipeline.");

    ModeB::Config config = {
        .inputDims = {
            .NBEAMS = 1,
            .NANTS  = 20,
            .NCHANS = 192,
            .NTIME  = 8192,
            .NPOLS  = 2,
        },
        .channelizerRate = 4,
        .beamformerBeams = 16,
        .castBlockSize = 512,
        .channelizerBlockSize = 512,
        .beamformerBlockSize = 512,
    };

    // Instantiate swapchain instance A and B.
    std::vector<std::unique_ptr<ModeB>> swapchain;
    swapchain.push_back(std::make_unique<ModeB>(config));
    swapchain.push_back(std::make_unique<ModeB>(config));

    // Allocate and register input buffers.
    std::vector<std::vector<CI8>> input;
    input.resize(swapchain.size());

    for (std::size_t i = 0; i < swapchain.size(); i++) {
        input[i].resize(swapchain[i]->getInputSize());
        swapchain[i]->pinBuffer(input[i], RegisterKind::ReadOnly);
    }

    // Allocate and register output buffers.
    std::vector<std::vector<CF16>> output;
    output.resize(swapchain.size());

    for (std::size_t i = 0; i < swapchain.size(); i++) {
        output[i].resize(swapchain[i]->getOutputSize());
        swapchain[i]->pinBuffer(output[i], RegisterKind::Default);
    }

    // Gather resources reports from instances.
    for (auto& instance : swapchain) {
        manager.save(*instance);
    }
    manager.report();

    // Repeat each operation 150 times to average out the execution time.
    auto t1 = high_resolution_clock::now();
    for (int i = 0; i < 150; i++) {
        // Upload the data of both instances in parallel.
        for (std::size_t i = 0; i < swapchain.size(); i++) {
            if (swapchain[i]->upload(input[i]) != Result::SUCCESS) {
                BL_WARN("Can't upload data. Test is exiting...");
                return 1;
            }
        }

        // Process the data of both instances in parallel.
        for (std::size_t i = 0; i < swapchain.size(); i++) {
            if (swapchain[i]->process() != Result::SUCCESS) {
                BL_WARN("Can't process data. Test is exiting...");
                return 1;
            }
        }

        // Download the data of both instances in parallel.
        for (std::size_t i = 0; i < swapchain.size(); i++) {
            if (swapchain[i]->download(output[i]) != Result::SUCCESS) {
                BL_WARN("Can't download data. Test is exiting...");
                return 1;
            }
        }

        // Wait for both instances to finish.
        cudaDeviceSynchronize();
    }
    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = (t2 - t1) / 300.0;
    BL_INFO("Cycle time average was {} ms.", ms_double.count());

    BL_INFO("Test succeeded.");

    return 0;
}
