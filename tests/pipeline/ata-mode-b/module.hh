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

class Module : public Pipeline {
 public:
    struct Config {
        ArrayDims inputDims;
        std::size_t channelizerRate = 4;
        std::size_t beamformerBeams = 16;
        std::size_t castBlockSize = 512;
        std::size_t channelizerBlockSize = 512;
        std::size_t beamformerBlockSize = 512;
    };

    explicit Module(const Config& configuration) : config(configuration) {
        if (this->setup() != Result::SUCCESS) {
            throw Result::ERROR;
        }
    }

    std::size_t getInputSize() const {
        return channelizer->getBufferSize();
    }

    std::size_t getOutputSize() const {
        return beamformer->getOutputSize();
    }

    Result run(const std::span<CI8>& in, std::span<CF16>& out, bool async = true) {
        this->input = in;
        this->output = out;
        return this->loop(async);
    }

 protected:
    Result setupModules() final {
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

    Result setupMemory() final {
        BL_INFO("Allocating resources.");

        BL_CHECK(allocateBuffer(phasors, beamformer->getPhasorsSize()));
        BL_CHECK(allocateBuffer(bufferA, channelizer->getBufferSize()));
        BL_CHECK(allocateBuffer(bufferB, channelizer->getBufferSize()));
        BL_CHECK(allocateBuffer(bufferC, channelizer->getBufferSize()));
        BL_CHECK(allocateBuffer(bufferD, beamformer->getOutputSize()));
        BL_CHECK(allocateBuffer(bufferE, beamformer->getOutputSize()));

        return Result::SUCCESS;
    }

    Result setupReport(Resources& res) final {
        BL_INFO("Reporting resources.");

        res.transfer.h2d += bufferA.size_bytes();
        res.transfer.d2h += bufferE.size_bytes();

        return Result::SUCCESS;
    }

    Result loopUpload() final {
        BL_CHECK(this->copyBuffer(bufferA, input, CopyKind::H2D));

        return Result::SUCCESS;
    }

    Result loopProcess(cudaStream_t& cudaStream) final {
        BL_CHECK(cast->run(bufferA, bufferB, cudaStream));
        BL_CHECK(channelizer->run(bufferB, bufferC, cudaStream));
        BL_CHECK(beamformer->run(bufferC, phasors, bufferD, cudaStream));
        BL_CHECK(cast->run(bufferD, bufferE, cudaStream));

        return Result::SUCCESS;
    }

    Result loopDownload() final {
        BL_CHECK(this->copyBuffer(output, bufferE, CopyKind::D2H));

        return Result::SUCCESS;
    }

 private:
    const Config& config;

    std::span<CI8> input;
    std::span<CF16> output;

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
