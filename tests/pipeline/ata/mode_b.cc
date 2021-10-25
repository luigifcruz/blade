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
    struct ConfigInternal {
        std::size_t channelizerRate = 4;
        std::size_t beamformerBeams = 16;
        std::size_t castBlockSize = 512;
        std::size_t channelizerBlockSize = 512;
        std::size_t beamformerBlockSize = 350;
    };

    struct Config : ArrayDims, ConfigInternal {};

    ModeB(const Config & configuration) : config(configuration) {
        cast = Factory<Cast::Generic>({
            .blockSize = config.castBlockSize,
        });

        channelizer = Factory<Channelizer::Generic>({
            config,
            {
                .fftSize = config.channelizerRate,
                .blockSize = config.channelizerBlockSize,
            },
        });

        beamformer = Factory<Beamformer::ATA>({
            channelizer->getOutputDims(config.beamformerBeams),
            {
                .blockSize = config.beamformerBlockSize,
            }
        });

        if (this->commit() != Result::SUCCESS) {
            throw Result::ERROR;
        }
    }

    ~ModeB() {
        Free(bufferA);
        Free(bufferB);
        Free(bufferC);
        Free(bufferD);
        Free(phasors);
    }

    std::size_t getInputSize() const {
        return channelizer->getBufferSize();
    }

    std::size_t getOutputSize() const {
        return beamformer->getOutputSize();
    }

    Result upload(const std::span<std::complex<int8_t>> &input) {
        BL_CHECK(Transfer(bufferA, input, CopyKind::H2D));

        return Result::SUCCESS;
    }

    Result download(std::span<std::complex<half>> output) {
        BL_CHECK(Transfer(output, bufferE, CopyKind::D2H));

        return Result::SUCCESS;
    }

    Resources getResources() {
        Resources res;

        // Report device memory.
        res.memory.device += phasors.size_bytes();
        res.memory.device += bufferA.size_bytes();
        res.memory.device += bufferB.size_bytes();
        res.memory.device += bufferC.size_bytes();
        res.memory.device += bufferD.size_bytes();
        res.memory.device += bufferE.size_bytes();

        // Report host memory.
        res.memory.host += phasors.size_bytes();
        res.memory.host += bufferA.size_bytes();
        res.memory.host += bufferE.size_bytes();

        // Report transfers.
        res.transfer.h2d += bufferA.size_bytes();
        res.transfer.d2h += bufferE.size_bytes();

        return res;
    }

protected:
    Result underlyingAllocate() {
        BL_CHECK(Allocate(beamformer->getPhasorsSize(), phasors));
        BL_CHECK(Allocate(channelizer->getBufferSize(), bufferA));
        BL_CHECK(Allocate(channelizer->getBufferSize(), bufferB));
        BL_CHECK(Allocate(channelizer->getBufferSize(), bufferC));
        BL_CHECK(Allocate(beamformer->getOutputSize(), bufferD));
        BL_CHECK(Allocate(beamformer->getOutputSize(), bufferE));

        return Result::SUCCESS;
    }

    Result underlyingProcess(cudaStream_t & cudaStream) {
        BL_CHECK(cast->run(bufferA, bufferB, cudaStream));
        BL_CHECK(channelizer->run(bufferB, bufferC, cudaStream));
        BL_CHECK(beamformer->run(bufferC, phasors, bufferD, cudaStream));
        BL_CHECK(cast->run(bufferD, bufferE, cudaStream));

        return Result::SUCCESS;
    }

private:
    const Config& config;

    std::span<std::complex<float>> phasors;
    std::span<std::complex<int8_t>> bufferA;
    std::span<std::complex<float>> bufferB;
    std::span<std::complex<float>> bufferC;
    std::span<std::complex<float>> bufferD;
    std::span<std::complex<half>> bufferE;

    std::unique_ptr<Cast::Generic> cast;
    std::unique_ptr<Beamformer::ATA> beamformer;
    std::unique_ptr<Channelizer::Generic> channelizer;
};

int main() {
    Logger guard{};
    Manager manager{};

    BL_INFO("Testing ATA beamformer pipeline.");

    ModeB::Config config = {
        {
            .NBEAMS = 1,
            .NANTS  = 20,
            .NCHANS = 96,
            .NTIME  = 35000,
            .NPOLS  = 2,
        }, {
            .channelizerRate = 4,
            .beamformerBeams = 16,
            .castBlockSize = 512,
            .channelizerBlockSize = 512,
            .beamformerBlockSize = 350,
        },
    };

    // Instantiate swapchain instance A and B.
    std::vector<std::unique_ptr<ModeB>> swapchain;
    swapchain.push_back(std::make_unique<ModeB>(config));
    swapchain.push_back(std::make_unique<ModeB>(config));

    // Gather resources reports from instances.
    for (auto & instance : swapchain) {
        manager.save(*instance);
    }
    manager.report();

    // Allocate and register input buffers.
    std::vector<std::vector<std::complex<int8_t>>> input;
    input.resize(swapchain.size());

    for (std::size_t i = 0; i < swapchain.size(); i++) {
        input[i].resize(swapchain[i]->getInputSize());
        swapchain[i]->Register(input[i], RegisterKind::ReadOnly);
    }

    // Allocate and register output buffers.
    std::vector<std::vector<std::complex<half>>> output;
    output.resize(swapchain.size());

    for (std::size_t i = 0; i < swapchain.size(); i++) {
        output[i].resize(swapchain[i]->getOutputSize());
        swapchain[i]->Register(output[i], RegisterKind::Default);
    }

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
