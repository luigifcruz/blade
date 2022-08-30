#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#include "mode_bh.h"

typedef struct {
    size_t step_count;

    size_t input_buffer_count;
    size_t output_buffer_count;

    void** input_buffers;
    void** output_buffers;
} userdata_t;

// Gives Blade::Runner an empty input buffer when asked.
bool input_buffer_fetch_cb(void* user_data_ptr, void** buffer) {
    userdata_t* user_data = (userdata_t*)user_data_ptr;

    if (user_data->step_count >= BLADE_ATA_MODE_BH_ITERATIONS) {
        return false;
    }

    *buffer = user_data->input_buffers[user_data->input_buffer_count]; 

    user_data->input_buffer_count = (user_data->input_buffer_count + 1) 
        % BLADE_ATA_MODE_BH_NUMBER_OF_WORKERS;

    printf("Input fetch: %ld %p\n", user_data->step_count, (void*)*buffer);

    user_data->step_count++;

    return true;
}

// Recycle an input buffer for next iteration.
void input_buffer_ready_cb(void* user_data_ptr, const void* buffer) {
    userdata_t* user_data = (userdata_t*)user_data_ptr;

    printf("Input ready: %ld %p\n", user_data->step_count, (void*)buffer);
}

// Gives Blade::Runner an empty output buffer when asked.
bool output_buffer_fetch_cb(void* user_data_ptr, void** buffer) {
    userdata_t* user_data = (userdata_t*)user_data_ptr;

    *buffer = user_data->output_buffers[user_data->output_buffer_count]; 

    user_data->output_buffer_count = (user_data->output_buffer_count + 1) 
        % BLADE_ATA_MODE_BH_NUMBER_OF_WORKERS;

    printf("Output fetch: %ld %p\n", user_data->step_count, (void*)*buffer);

    return true;
}

// Consume an output buffer and recycle it for the next iteration.
void output_buffer_ready_cb(void* user_data_ptr, const void* buffer) {
    userdata_t* user_data = (userdata_t*)user_data_ptr;

    printf("Output ready: %ld %p\n", user_data->step_count, (void*)buffer);
}

int main(int argc, char **argv) {
    if (argc == 3 && !blade_use_device(atoi(argv[2]))) {
        printf("failed to set device\n");
        return 1;
    }

    blade_ata_bh_initialize(BLADE_ATA_MODE_BH_NUMBER_OF_WORKERS);

    userdata_t user_data = {
        .step_count = 0,
        .input_buffer_count = 0,
        .output_buffer_count = 0,
        .input_buffers = (void**)malloc(BLADE_ATA_MODE_BH_NUMBER_OF_WORKERS * sizeof(void*)),
        .output_buffers = (void**)malloc(BLADE_ATA_MODE_BH_NUMBER_OF_WORKERS * sizeof(void*)),
    };

    for (int i = 0; i < BLADE_ATA_MODE_BH_NUMBER_OF_WORKERS; i++) {
        size_t input_byte_size = blade_ata_bh_get_input_size() * sizeof(int8_t) * 2;
        user_data.input_buffers[i] = (void*)malloc(input_byte_size);
        blade_pin_memory(user_data.input_buffers[i], input_byte_size);

        size_t output_byte_size = blade_ata_bh_get_output_size() * sizeof(float);
        user_data.output_buffers[i] = (void*)malloc(output_byte_size);
        blade_pin_memory(user_data.output_buffers[i], output_byte_size);
    }

    blade_ata_bh_register_user_data(&user_data);
    blade_ata_bh_register_input_buffer_fetch_cb(&input_buffer_fetch_cb);
    blade_ata_bh_register_input_buffer_ready_cb(&input_buffer_ready_cb);
    blade_ata_bh_register_output_buffer_fetch_cb(&output_buffer_fetch_cb);
    blade_ata_bh_register_output_buffer_ready_cb(&output_buffer_ready_cb);

    clock_t begin = clock();

    // Blade is single-threaded!
    // For the love of god, don't put this on a different thread.
    while (blade_ata_bh_compute_step());
    printf("Finished processing %ld steps.\n", user_data.step_count);

    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Example finished in %lf s.\n", time_spent);
    double iteration_time = (time_spent / (BLADE_ATA_MODE_BH_ACCUMULATE_RATE * 8)) * 1000;
    printf("Average execution per-iteration: %lf ms.\n", iteration_time);

    blade_ata_bh_terminate();

    for (int i = 0; i < BLADE_ATA_MODE_BH_NUMBER_OF_WORKERS; i++) {
        free(user_data.input_buffers[i]);
        free(user_data.output_buffers[i]);
    }

    free(user_data.input_buffers);
    free(user_data.output_buffers);
}
