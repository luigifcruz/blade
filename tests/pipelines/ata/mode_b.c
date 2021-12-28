#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "blade/pipelines/ata/mode_b.h"

#define SYNC_MODE 0

int main(int argc, char **argv) {
    if (argc == 3 && !blade_use_device(atoi(argv[2]))) {
        printf("failed to set device\n");
        return 1;
    }

    size_t number_of_workers = 2;
    blade_module_t mod = blade_ata_b_initialize(number_of_workers);

    void** input_buffers = (void**)malloc(number_of_workers * sizeof(void*));
    void** output_buffers = (void**)malloc(number_of_workers * sizeof(void*));

    for (int i = 0; i < number_of_workers; i++) {
        size_t input_byte_size = blade_ata_b_get_input_size(mod) * sizeof(int8_t) * 2;
        input_buffers[i] = (void*)malloc(input_byte_size);
        blade_pin_memory(input_buffers[i], input_byte_size);

        size_t output_byte_size = blade_ata_b_get_output_size(mod) * sizeof(int16_t) * 2;
        output_buffers[i] = (void*)malloc(output_byte_size);
        blade_pin_memory(output_buffers[i], output_byte_size);
    }

#if SYNC_MODE

    for (int i = 0; i < 255; i++) {
        blade_ata_b_process(mod, input_buffers, output_buffers);
    }

#else
    int h = 0;

    for (int i = 0; i < 510; i++) {
        if (blade_ata_b_enqueue(mod, input_buffers[h], output_buffers[h])) {
            h = (h + 1) % number_of_workers;
        }

        if (blade_ata_b_dequeue(mod, NULL, NULL)) {
            // consume pointer
        }
    }

#endif

    blade_ata_b_terminate(mod);

    for (int i = 0; i < number_of_workers; i++) {
        free(input_buffers[i]);
        free(output_buffers[i]);
    }

    free(input_buffers);
    free(output_buffers);
}
