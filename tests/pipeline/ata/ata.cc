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

    ModeB pipeline({
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
    });

    ModeB pipeline2({
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
    });

    manager.save(pipeline);
    manager.save(pipeline2);
    manager.report();

    std::vector<std::complex<int8_t>> input(pipeline.getInputSize());
    std::vector<std::complex<half>> output(pipeline.getOutputSize());

    pipeline.Register(input, RegisterKind::ReadOnly);
    pipeline.Register(output, RegisterKind::Default);

    std::vector<std::complex<int8_t>> input2(pipeline2.getInputSize());
    std::vector<std::complex<half>> output2(pipeline2.getOutputSize());

    pipeline2.Register(input2, RegisterKind::ReadOnly);
    pipeline2.Register(output2, RegisterKind::Default);

    auto t1 = high_resolution_clock::now();
    for (int i = 0; i < 150; i++) {
        if (pipeline.upload(input) != Result::SUCCESS) {
            BL_WARN("Can't upload data. Test is exiting...");
            return 1;
        }

        if (pipeline2.upload(input2) != Result::SUCCESS) {
            BL_WARN("Can't upload data. Test is exiting...");
            return 1;
        }

        if (pipeline.process() != Result::SUCCESS) {
            BL_WARN("Can't process data. Test is exiting...");
            return 1;
        }

        if (pipeline2.process() != Result::SUCCESS) {
            BL_WARN("Can't process data. Test is exiting...");
            return 1;
        }

        if (pipeline.download(output) != Result::SUCCESS) {
            BL_WARN("Can't download data. Test is exiting...");
            return 1;
        }

        if (pipeline2.download(output2) != Result::SUCCESS) {
            BL_WARN("Can't download data. Test is exiting...");
            return 1;
        }

        cudaDeviceSynchronize();
    }
    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = (t2 - t1) / 300.0;
    BL_INFO("Cycle time average was {} ms.", ms_double.count());

    BL_INFO("Test succeeded.");

    return 0;
}
