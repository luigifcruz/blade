#include "mode_bh_stub.h"

typedef struct {
    size_t step_count;

    size_t input_buffer_count;
    size_t output_buffer_count;

    void** input_buffers;
    void** output_buffers;
} userdata_t;

static userdata_t user_data = {
    .step_count = 0,
    .input_buffer_count = 0,
    .output_buffer_count = 0,
    .input_buffers = NULL,
    .output_buffers = NULL,
};

// Gives Blade::Runner an empty input buffer when asked.
bool input_buffer_fetch_cb(void* user_data_ptr, void** buffer) {
    userdata_t* user_data = (userdata_t*)user_data_ptr;

    *buffer = user_data->input_buffers[user_data->input_buffer_count]; 

    user_data->input_buffer_count = (user_data->input_buffer_count + 1) 
        % BLADE_ATA_MODE_BH_NUMBER_OF_WORKERS;

#ifdef DEBUG
    printf("Input fetch: %ld %p\n", user_data->step_count, (void*)*buffer);
#endif

    user_data->step_count++;

    return true;
}

// Recycle an input buffer for next iteration.
void input_buffer_ready_cb(void* user_data_ptr, const void* buffer) {
#ifdef DEBUG
    userdata_t* user_data = (userdata_t*)user_data_ptr;
    printf("Input ready: %ld %p\n", user_data->step_count, (void*)buffer);
#endif
}

// Gives Blade::Runner an empty output buffer when asked.
bool output_buffer_fetch_cb(void* user_data_ptr, void** buffer) {
    userdata_t* user_data = (userdata_t*)user_data_ptr;

    *buffer = user_data->output_buffers[user_data->output_buffer_count]; 

    user_data->output_buffer_count = (user_data->output_buffer_count + 1) 
        % BLADE_ATA_MODE_BH_NUMBER_OF_WORKERS;

#ifdef DEBUG
    printf("Output fetch: %ld %p\n", user_data->step_count, (void*)*buffer);
#endif

    return true;
}

// Consume an output buffer and recycle it for the next iteration.
void output_buffer_ready_cb(void* user_data_ptr, const void* buffer) {
#ifdef DEBUG
    userdata_t* user_data = (userdata_t*)user_data_ptr;
    printf("Output ready: %ld %p\n", user_data->step_count, (void*)buffer);
#endif
}

int mode_bh_setup() {
    blade_ata_bh_initialize(BLADE_ATA_MODE_BH_NUMBER_OF_WORKERS);

    user_data.input_buffers = (void**)malloc(BLADE_ATA_MODE_BH_NUMBER_OF_WORKERS * sizeof(void*));
    user_data.output_buffers = (void**)malloc(BLADE_ATA_MODE_BH_NUMBER_OF_WORKERS * sizeof(void*));

    for (int i = 0; i < BLADE_ATA_MODE_BH_NUMBER_OF_WORKERS; i++) {
        size_t input_byte_size = blade_ata_bh_get_input_size() * sizeof(int8_t) * 2;
        user_data.input_buffers[i] = (void*)malloc(input_byte_size);
        cudaHostRegister(user_data.input_buffers[i], input_byte_size, cudaHostRegisterDefault);

        size_t output_byte_size = blade_ata_bh_get_output_size() * sizeof(float);
        user_data.output_buffers[i] = (void*)malloc(output_byte_size);
        cudaHostRegister(user_data.output_buffers[i], output_byte_size, cudaHostRegisterDefault);
    }

    blade_ata_bh_register_user_data(&user_data);
    blade_ata_bh_register_input_buffer_fetch_cb(&input_buffer_fetch_cb);
    blade_ata_bh_register_input_buffer_ready_cb(&input_buffer_ready_cb);
    blade_ata_bh_register_output_buffer_fetch_cb(&output_buffer_fetch_cb);
    blade_ata_bh_register_output_buffer_ready_cb(&output_buffer_ready_cb);

    return 0;
}

int mode_bh_loop(int iterations) {
    // Blade is single-threaded!
    // For the love of god, don't put this on a different thread.
    while (user_data.step_count < iterations) {
        blade_ata_bh_compute_step();
    }

    user_data.step_count = 0;

    return 0;
}

int mode_bh_terminate() {
    blade_ata_bh_terminate();

    for (int i = 0; i < BLADE_ATA_MODE_BH_NUMBER_OF_WORKERS; i++) {
        free(user_data.input_buffers[i]);
        free(user_data.output_buffers[i]);
    }

    free(user_data.input_buffers);
    free(user_data.output_buffers);

    return 0;
}
