#include "module.hh"

extern "C" {
#include "module.h"
}

struct State {
    Module::Config config = {
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

    std::unique_ptr<Logger> guard;
    std::unique_ptr<Manager> manager;
    std::vector<std::unique_ptr<Module>> swapchain;

    std::size_t runs = 0;
    time_point<system_clock, duration<double, std::milli>> t1;
};

module_t init(size_t batch_size) {
    auto self = new State();

    if (batch_size < 1) {
        BL_FATAL("Batch size should be larger than zero.");
        return nullptr;
    }

    // Instantiate modules.
    self->guard = std::make_unique<Logger>();
    self->manager = std::make_unique<Manager>();

    // Logging ready.
    BL_INFO("Pipeline for ATA Mode B started.");

    // Instantiate swapchain workers.
    for (std::size_t i = 0; i < batch_size; i++) {
        self->swapchain.push_back(std::make_unique<Module>(self->config));
    }

    // Register resources.
    for (auto& worker : self->swapchain) {
        self->manager->save(*worker);
    }
    self->manager->report();

    return self;
}

void deinit(module_t mod) {
    auto self = static_cast<State*>(mod);

    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> elapsed = (t2 - self->t1);

    BL_INFO("Cycle time average was {} ms for each of {} runs.",
        elapsed.count() / self->runs, self->runs);
    BL_INFO("Pipeline exiting.");

    delete self;
}

int pin_memory(module_t mod, void* buffer, size_t size) {
    return cudaHostRegister(buffer, size, cudaHostRegisterDefault);
}

size_t get_input_size(module_t mod) {
    auto self = static_cast<State*>(mod);
    return self->swapchain[0]->getInputSize();
}

size_t get_output_size(module_t mod) {
    auto self = static_cast<State*>(mod);
    return self->swapchain[0]->getOutputSize();
}

int process(module_t mod, void** input, void** output) {
    auto self = static_cast<State*>(mod);

    // Disregard first two iterations.
    if (self->runs == 2 * self->swapchain.size()) {
        self->t1 = high_resolution_clock::now();
    }

    // Process the data of both instances in parallel.
    for (std::size_t i = 0; i < self->swapchain.size(); i++) {
        auto& worker = self->swapchain[i];

        auto ibuf = static_cast<CI8*>(input[i]);
        auto in = std::span(ibuf, worker->getInputSize());

        auto obuf = static_cast<CF16*>(output[i]);
        auto out = std::span(obuf, worker->getOutputSize());

        if (worker->run(in, out) != Result::SUCCESS) {
            BL_WARN("Can't process data. Test is exiting.:q..");
            return 1;
        }
    }

    // Wait for both instances to finish.
    cudaDeviceSynchronize();
    self->runs += self->swapchain.size();

    return to_underlying(Result::SUCCESS);
}
