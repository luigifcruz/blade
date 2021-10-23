#include <chrono>

#include "blade/cast/base.hh"
#include "blade/checker/base.hh"
#include "blade/channelizer/base.hh"
#include "blade/beamformer/ata.hh"
#include "blade/manager.hh"

using namespace Blade;
using namespace std::chrono;

class Pipeline {
public:
    struct ConfigInternal {
        std::size_t channelizerRate = 4;
        std::size_t beamformerBeams = 16;
        std::size_t castBlockSize = 512;
        std::size_t channelizerBlockSize = 512;
        std::size_t beamformerBlockSize = 350;
    };

    struct Config : ArrayDims, ConfigInternal {};

    Pipeline(const Config & configuration) : config(configuration) {
        BL_INFO("Initializing modules...")
        inputCast = Factory<Cast::Generic>({
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

        manager = Factory<Manager>({
            .pcie_bw = static_cast<size_t>(11e9),
        });

        BL_INFO("Allocating device memory...")
        if (this->allocate() != Result::SUCCESS) {
            BL_FATAL("Pipeline initialization failed. Exiting...");
            throw Result::ERROR;
        }

        BL_INFO("Generating CUDA Graph...")
        if (this->generateGraph() != Result::SUCCESS) {
            BL_FATAL("Pipeline initialization failed. Exiting...");
            throw Result::ERROR;
        }

        BL_INFO("Initialization finished.");
    }

    ~Pipeline() {
        cudaFree(bufferA.data());
        cudaFree(bufferB.data());
        cudaFree(bufferC.data());
        cudaFree(bufferD.data());
        cudaFree(phasors.data());
    }

    std::size_t getInputSize() const {
        return channelizer->getBufferSize();
    }

    std::size_t getOutputSize() const {
        return beamformer->getOutputSize();
    }

    Result upload(const std::span<std::complex<int8_t>> &input) {
        BL_CUDA_CHECK(cudaMemcpyAsync(bufferA.data(), input.data(),
                    bufferA.size_bytes(), cudaMemcpyHostToDevice, cudaStream), [&]{
            BL_FATAL("Can't copy data from host to device: {}", err);
        });

        return Result::SUCCESS;
    }

    Result process(bool waitCompletion = false) {
        BL_CUDA_CHECK(cudaGraphLaunch(instance, 0), [&]{
            BL_FATAL("Failed launch CUDA graph: {}", err);
        });

        if (waitCompletion) {
            cudaDeviceSynchronize();
        }

        return Result::SUCCESS;
    }

    Result download(std::span<std::complex<float>> output) {
        BL_CUDA_CHECK(cudaMemcpyAsync(output.data(), bufferD.data(),
                    bufferD.size_bytes(), cudaMemcpyDeviceToHost, cudaStream), [&]{
            BL_FATAL("Can't copy data from device to host: {}", err);
        });

        return Result::SUCCESS;
    }

private:
    const Config& config;

    cudaGraph_t graph;
    cudaStream_t cudaStream;
    cudaGraphExec_t instance;

    std::span<std::complex<float>> phasors;
    std::span<std::complex<int8_t>> bufferA;
    std::span<std::complex<float>> bufferB;
    std::span<std::complex<float>> bufferC;
    std::span<std::complex<float>> bufferD;

    std::unique_ptr<Manager> manager;
    std::unique_ptr<Cast::Generic> inputCast;
    std::unique_ptr<Beamformer::ATA> beamformer;
    std::unique_ptr<Channelizer::Generic> channelizer;

    Result allocate() {
        std::complex<float>* pphasors;
        std::complex<int8_t>* pbufferA;
        std::complex<float>* pbufferB;
        std::complex<float>* pbufferC;
        std::complex<float>* pbufferD;

        std::size_t sphasors = beamformer->getPhasorsSize() * sizeof(pphasors[0]);
        std::size_t sbufferA = channelizer->getBufferSize() * sizeof(pbufferA[0]);
        std::size_t sbufferB = channelizer->getBufferSize() * sizeof(pbufferB[0]);
        std::size_t sbufferC = channelizer->getBufferSize() * sizeof(pbufferC[0]);
        std::size_t sbufferD = beamformer->getOutputSize() * sizeof(pbufferD[0]);

        BL_CUDA_CHECK(cudaMalloc(&pphasors, sphasors), [&]{
            BL_FATAL("Can't allocate phasor buffer: {}", err);
        });

        BL_CUDA_CHECK(cudaMalloc(&pbufferA, sbufferA), [&]{
            BL_FATAL("Can't allocate buffer A: {}", err);
        });

        BL_CUDA_CHECK(cudaMalloc(&pbufferB, sbufferB), [&]{
            BL_FATAL("Can't allocate buffer B: {}", err);
        });

        BL_CUDA_CHECK(cudaMalloc(&pbufferC, sbufferC), [&]{
            BL_FATAL("Can't allocate buffer C: {}", err);
        });

        BL_CUDA_CHECK(cudaMalloc(&pbufferD, sbufferD), [&]{
            BL_FATAL("Can't allocate buffer D: {}", err);
        });

        this->phasors = std::span(pphasors, beamformer->getPhasorsSize());
        this->bufferA = std::span(pbufferA, channelizer->getBufferSize());
        this->bufferB = std::span(pbufferB, channelizer->getBufferSize());
        this->bufferC = std::span(pbufferC, channelizer->getBufferSize());
        this->bufferD = std::span(pbufferD, beamformer->getOutputSize());

        manager->save({
            .memory = {
                .device = sphasors + sbufferA + sbufferB + sbufferC + sbufferD,
                .host = sphasors + sbufferA + sbufferD,
            },
            .transfer = {
                .d2h = sbufferD,
                .h2d = sbufferA,
            }
        }).report();

        return Result::SUCCESS;
    }

    Result runKernels() {
        BL_CHECK(inputCast->run(bufferA, bufferB, cudaStream));
        BL_CHECK(channelizer->run(bufferB, bufferC, cudaStream));
        BL_CHECK(beamformer->run(bufferC, phasors, bufferD, cudaStream));

        return Result::SUCCESS;
    }

    Result generateGraph() {
        BL_CUDA_CHECK(cudaStreamCreateWithFlags(&cudaStream,
                    cudaStreamNonBlocking), [&]{
            BL_FATAL("Failed to create stream for CUDA Graph: {}", err);
        });

        runKernels(); // Run kernels once to populate cache.

        BL_CUDA_CHECK(cudaStreamBeginCapture(cudaStream,
                    cudaStreamCaptureModeGlobal), [&]{
            BL_FATAL("Failed to begin the capture of CUDA Graph: {}", err);
        });

        runKernels();

        BL_CUDA_CHECK(cudaStreamEndCapture(cudaStream, &graph), [&]{
            BL_FATAL("Failed to end the capture of CUDA Graph: {}", err);
        });

        BL_CUDA_CHECK(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0), [&]{
            BL_FATAL("Failed to instantiate CUDA Graph: {}", err);
        });

        return Result::SUCCESS;
    }
};

int main() {
    Logger guard{};

    BL_INFO("Testing ATA beamformer pipeline.");

    Pipeline pipeline({
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

    Pipeline pipeline2({
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

    std::vector<std::complex<int8_t>> input(pipeline.getInputSize());
    std::vector<std::complex<float>> output(pipeline.getOutputSize());

    cudaHostRegister(input.data(), input.size() * sizeof(input[0]), cudaHostRegisterReadOnly);
    cudaHostRegister(output.data(), output.size() * sizeof(output[0]), cudaHostRegisterDefault);

    std::vector<std::complex<int8_t>> input2(pipeline2.getInputSize());
    std::vector<std::complex<float>> output2(pipeline2.getOutputSize());

    cudaHostRegister(input2.data(), input2.size() * sizeof(input2[0]), cudaHostRegisterReadOnly);
    cudaHostRegister(output2.data(), output2.size() * sizeof(output2[0]), cudaHostRegisterDefault);

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
