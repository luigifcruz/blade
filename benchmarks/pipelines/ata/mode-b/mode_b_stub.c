#include "mode_b_stub.h"

static void** input_buffers;
static void** output_buffers;

int mode_b_init() {
    blade_ata_b_initialize(BLADE_ATA_MODE_B_NUMBER_OF_WORKERS);

    return 0;
}

int mode_b_setup() {
    input_buffers = (void**)malloc(BLADE_ATA_MODE_B_NUMBER_OF_WORKERS * sizeof(void*));
    output_buffers = (void**)malloc(BLADE_ATA_MODE_B_NUMBER_OF_WORKERS * sizeof(void*));

    for (int i = 0; i < BLADE_ATA_MODE_B_NUMBER_OF_WORKERS; i++) {
        size_t input_byte_size = blade_ata_b_get_input_size() * sizeof(int8_t) * 2;
        input_buffers[i] = (void*)malloc(input_byte_size);
        cudaHostRegister(input_buffers[i], input_byte_size, cudaHostRegisterDefault);

        size_t output_byte_size = blade_ata_b_get_output_size() * BLADE_ATA_MODE_B_OUTPUT_NCOMPLEX_BYTES;
        output_buffers[i] = (void*)malloc(output_byte_size);
        cudaHostRegister(output_buffers[i], output_byte_size, cudaHostRegisterDefault);
    }

    return 0;
}

int mode_b_loop(int iterations) {
    int h = 0, i = 0;

    while (i < iterations) {
        if (blade_ata_b_enqueue(input_buffers[h], output_buffers[h], i)) {
            h = (h + 1) % BLADE_ATA_MODE_B_NUMBER_OF_WORKERS;
        }

        size_t id;
        if (blade_ata_b_dequeue(&id)) {
#ifdef DEBUG
            printf("Task %zu finished.\n", id);
#endif
            i++;
        }
    }

    return 0;
}

int mode_b_terminate() {
    blade_ata_b_terminate();

    for (int i = 0; i < BLADE_ATA_MODE_B_NUMBER_OF_WORKERS; i++) {
        free(input_buffers[i]);
        free(output_buffers[i]);
    }

    free(input_buffers);
    free(output_buffers);

    return 0;
}
