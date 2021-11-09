#include "blade/pipelines/ata/mode_b.hh"

extern "C" {
    #include "blade/pipelines/ata/mode_b.h"
}

using namespace Blade;
using namespace Blade::Pipelines::ATA;

struct State {
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

    struct Job {
        std::size_t index;
        void* input_ptr;
        void* output_ptr;
    };
    std::deque<Job> fifo;
    std::size_t head = 0;

    std::unique_ptr<Logger> guard;
    std::unique_ptr<Manager> manager;
    std::vector<std::unique_ptr<ModeB>> swapchain;

    std::size_t runs = 0;
    time_point<system_clock, duration<double, std::milli>> t1;
};

blade_module_t blade_ata_b_initialize(size_t number_of_workers) {
    auto self = new State();

    if (number_of_workers < 1) {
        BL_FATAL("Number of workers should be larger than zero.");
        return nullptr;
    }

    // Instantiate modules.
    self->guard = std::make_unique<Logger>();
    self->manager = std::make_unique<Manager>();

    // Logging ready.
    BL_INFO("Pipeline for ATA Mode B started.");

    // Instantiate swapchain workers.
    for (std::size_t i = 0; i < number_of_workers; i++) {
        self->swapchain.push_back(std::make_unique<ModeB>(self->config));
    }

    // Register resources.
    for (auto& worker : self->swapchain) {
        self->manager->save(worker->getResources());
    }
    self->manager->report();

    return self;
}

void blade_ata_b_terminate(blade_module_t mod) {
    auto self = static_cast<State*>(mod);

    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> elapsed = (t2 - self->t1);

    BL_INFO("Cycle time average was {} ms for each of {} runs.",
        elapsed.count() / self->runs, self->runs);
    BL_INFO("Pipeline exiting.");

    delete self;
}

size_t blade_ata_b_get_input_size(blade_module_t mod) {
    auto self = static_cast<State*>(mod);
    return self->swapchain[0]->getInputSize();
}

size_t blade_ata_b_get_output_size(blade_module_t mod) {
    auto self = static_cast<State*>(mod);
    return self->swapchain[0]->getOutputSize();
}

int process(blade_module_t mod, int idx, void* input, void* output) {
    auto self = static_cast<State*>(mod);

    if (self->runs == 2) {
        self->t1 = high_resolution_clock::now();
    }

    auto& worker = self->swapchain[idx];

    auto ibuf = static_cast<CI8*>(input);
    auto in = std::span(ibuf, worker->getInputSize());

    auto obuf = static_cast<CF16*>(output);
    auto out = std::span(obuf, worker->getOutputSize());

    if (worker->run(in, out) != Result::SUCCESS) {
        BL_WARN("Can't process data. Test is exiting...");
        return 1;
    }

    self->runs += 1;

    return to_underlying(Result::SUCCESS);
}

int blade_ata_b_process(blade_module_t mod, void** input, void** output) {
    auto self = static_cast<State*>(mod);

    // Process the data of both instances in parallel.
    for (std::size_t i = 0; i < self->swapchain.size(); i++) {
        process(mod, i, input[i], output[i]);
    }

    // Wait for both instances to finish.
    for (auto& worker : self->swapchain) {
        worker->synchronize();
    }

    return to_underlying(Result::SUCCESS);
}

bool blade_ata_b_enqueue(blade_module_t mod, void* input, void* output) {
    auto self = static_cast<State*>(mod);

    // If full, try again later.
    if (self->fifo.size() == self->swapchain.size()) {
        return false;
    }

    // Start processing.
    process(mod, self->head, input, output);

    // Push to the FIFO.
    self->fifo.push_back({
        .index = self->head,
        .input_ptr = input,
        .output_ptr = output,
    });

    self->head = (self->head + 1) % self->swapchain.size();

    return true;
}

bool blade_ata_b_dequeue(blade_module_t mod, void** input, void** output) {
    auto self = static_cast<State*>(mod);

    // If empty, try again later.
    if (self->fifo.size() == 0) {
        return false;
    }

    auto& job = self->fifo.front();
    auto& worker = self->swapchain[job.index];

    // If full, wait.
    if (self->fifo.size() == self->swapchain.size()) {
        worker->synchronize();
    }

    // If front synchronized, return.
    if (!worker->isSyncronized()) {
        return false;
    }

    if (input != NULL) {
        *input = job.input_ptr;
    }

    if (output != NULL) {
        *output = job.output_ptr;
    }

    self->fifo.pop_front();

    return true;
}
