#include "blade/base.hh"
#include "blade/runner.hh"
#include "blade/bundles/generic/mode_x.hh"

using namespace Blade;

using ModeX = Bundles::Generic::ModeX<CI8, CF32>;

template<typename IT, typename OT>
class ModeXRunner : public Runner {
 public:
    struct Config {
        ArrayShape inputShape;
        ArrayShape outputShape;
    };

    explicit ModeXRunner(const Config& config, U64 preCorrelatorStackerMultiplier, U64 correlatorIntegrationRate)
        : inputBuffer(config.inputShape),
          outputBuffer(config.outputShape)
    {
        ModeX::Config cfg = {
            .inputShape = config.inputShape,
            .outputShape = config.outputShape,
            .preCorrelatorStackerMultiplier = preCorrelatorStackerMultiplier,
            .correlatorIntegrationRate = correlatorIntegrationRate,
            .correlatorConjugateAntennaIndex = 0
        };
        this->connect(
            pipeline,
            cfg,
            {
                .buffer = inputBuffer
            }
        );
    }


    Result transferIn(const ArrayTensor<Device::CPU, IT>& cpuInputBuffer) {
        BL_CHECK(this->copy(inputBuffer, cpuInputBuffer));
        return Result::SUCCESS;
    }

    Result transferInSynchronised(const ArrayTensor<Device::CPU, IT>& cpuInputBuffer) {
        // BL_CHECK(this->copy(inputBuffer, cpuInputBuffer));
        // return synchroniseHead();

        if (inputBuffer[getHeadIndex()].size() != cpuInputBuffer.size()) {
            BL_FATAL("Size mismatch between source and destination ({}, {}).",
                    cpuInputBuffer.size(), inputBuffer[getHeadIndex()].size());
        }

        if (inputBuffer[getHeadIndex()].shape() != cpuInputBuffer.shape()) {
            BL_FATAL("Shape mismatch between source ({}) and destination ({}).",
                    cpuInputBuffer.shape(), inputBuffer[getHeadIndex()].shape());
        }
        

        BL_CUDA_CHECK(cudaMemcpy(inputBuffer[getHeadIndex()].data(), cpuInputBuffer.data(), cpuInputBuffer.size_bytes(),
                    cudaMemcpyHostToDevice), [&]{
            BL_FATAL("Can't copy data: {}", err);
            return Result::CUDA_ERROR;
        });
        return Result::SUCCESS;
    }

    Result transferResult() {
        BL_CHECK(this->copy(outputBuffer, pipeline->getOutputBuffer()));
        return Result::SUCCESS;
    }

    Result transferOut(ArrayTensor<Device::CPU, OT>& cpuOutputBuffer) {
        BL_CHECK(this->copy(cpuOutputBuffer, outputBuffer));
        return Result::SUCCESS;
    }

 private:
    std::shared_ptr<ModeX> pipeline;

    Duet<ArrayTensor<Device::CUDA, IT>> inputBuffer;
    Duet<ArrayTensor<Device::CUDA, OT>> outputBuffer;
};

int main() {
    using ModeXRunner = ModeXRunner<CI8, CF32>;

    // Configuring ModeXRunner. 
    //
    // This example will take I8 samples as input and produce F32 as output.
    // It will also perform a concatenation in the time samples dimension.

    const U64 nof_antennas = 28;
    const U64 nof_channels = 1;
    const U64 nof_samples = 32768;
    const U64 nof_polarizations = 2;
    const U64 nof_prechannelizer_gathers = 2;
    const U64 nof_integrations = 8192;

    ModeXRunner::Config config = {
        .inputShape = ArrayShape({ nof_antennas, nof_channels, nof_samples, nof_polarizations }),
        .outputShape = ArrayShape({ nof_antennas*(nof_antennas+1)/2, nof_samples*nof_prechannelizer_gathers, 1, nof_polarizations*nof_polarizations }),
    }; // 65536/(16e6) = 4.096 ms

    auto pipeline = std::make_shared<ModeXRunner>(config, nof_prechannelizer_gathers, nof_integrations);

    // Allocating buffers.
    //
    // The data will be stored in the CPU memory, but the pipeline will
    // transfer it to the GPU memory before processing. We need to allocate 
    // multiple buffers to allow the pipeline to process multiple batches
    // in parallel.

    std::vector<ArrayTensor<Device::CPU, CI8>> inputBuffer;
    std::vector<ArrayTensor<Device::CPU, CF32>> outputBuffer;

    for (U64 i = 0; i < pipeline->numberOfStreams(); i++) {
        inputBuffer.push_back(ArrayTensor<Device::CPU, CI8>(config.inputShape));
        outputBuffer.push_back(ArrayTensor<Device::CPU, CF32>(config.outputShape));
    }

    size_t index = 0;
    for (U64 a = 0; a < nof_antennas; a++) {
        for (U64 c = 0; c < nof_channels; c++) {
            for (U64 t = 0; t < nof_samples; t++) {
                for (U64 p = 0; p < nof_polarizations; p++) {
                    for (U64 i = 0; i < inputBuffer.size(); i++) {
                        if (a == 2) {
                            if (p==0)
                                inputBuffer[i][index] = std::complex<int8_t>(1, 0);
                            else
                                inputBuffer[i][index] = std::complex<int8_t>(0, 0);
                        } else if (a == 3) {
                            if (p==0)
                                inputBuffer[i][index] = std::complex<int8_t>(0, 0);
                            else
                                inputBuffer[i][index] = std::complex<int8_t>(1, 0);
                        } else if (a == 1) {
                            inputBuffer[i][index] = std::complex<int8_t>(1, 0);
                        } else {
                            inputBuffer[i][index] = std::complex<int8_t>(a*3, (t%250-125)+p+1);
                        }
                    }
                    index += 1;
                }
            }
        }
    }

    // Running the pipeline.
    //
    // The pipeline will process the data in two batches. This parallelism is
    // important because while one batch is being uploaded to the GPU, the
    // other one is being processed.
    //
    // The output of each batch will be printed to the console. The output of 
    // all batches should be the same because the input data is not changing.
    // Expect the output of each batch to be the input data repeated 10 times.

    
    U64 dequeueCount = 0;
    U64 enqueueCount = 0;

    U64 iterations = 0;
    const U64 totalIterations = 2;

    U64 auto_baseline_indices[nof_antennas];
    U64 auto_index = 0;
    for (U64 a = 0; a < nof_antennas; a++) {
        auto_baseline_indices[a] = auto_index;
        auto_index += nof_antennas-a;
    }

    while (iterations < totalIterations) {

        auto inputCallback = [&](){
            return pipeline->transferIn(inputBuffer[enqueueCount++ % inputBuffer.size()]);
        };
        auto resultCallback = [&](){
            return pipeline->transferResult();
        };
        auto outputCallback = [&](){
            return pipeline->transferOut(outputBuffer[dequeueCount++ % outputBuffer.size()]);
        };
        struct timespec timestamp_start, timestamp_stop;
      
        clock_gettime(CLOCK_MONOTONIC, &timestamp_start);
        pipeline->enqueue(inputCallback, resultCallback, outputCallback, enqueueCount % inputBuffer.size(), dequeueCount % outputBuffer.size());

        pipeline->dequeue([&](const U64& inputId, 
                              const U64& outputId,
                              const bool& didOutput){
            // BL_INFO("Input ID: {} | Output ID: {} | Did Output: {}", inputId, outputId, didOutput);
            if (didOutput) {
                // BL_INFO("Input:  {}", inputBuffer[inputId % 2])
                // BL_INFO("Output: {}", outputBuffer[outputId % 2]);
                for (U64 a = 0; a < 4; a++) {
                    BL_INFO(
                        "Auto#{} (@{})",
                        a,
                        auto_baseline_indices[a]
                    );
                    for (U64 p = 0; p < 4; p++) {
                        BL_INFO(
                            "\n[C=0,T=0,P={}] ({} + j{})",
                            p,
                            outputBuffer[outputId][
                                auto_baseline_indices[a],
                                0,
                                0,
                                p
                            ].real(),
                            outputBuffer[outputId][
                                auto_baseline_indices[a],
                                0,
                                0,
                                p
                            ].imag()
                        );
                    }
                }
                iterations++;
            }
            return Result::SUCCESS;
        });
        clock_gettime(CLOCK_MONOTONIC, &timestamp_stop);
        int64_t ns_elapsed = (((int64_t)timestamp_stop.tv_sec-timestamp_start.tv_sec)*1000000000+(timestamp_stop.tv_nsec-timestamp_start.tv_nsec));
        // BL_INFO("#{} (e: {}, d: {}): {} ns", iterations, enqueueCount, dequeueCount, ns_elapsed);
    }

    BL_INFO("Example pipeline finished.");

    return 0;
}