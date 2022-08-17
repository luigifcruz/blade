#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#include "mode_bh.h"

int main(int argc, char **argv) {
    if (argc == 3 && !blade_use_device(atoi(argv[2]))) {
        printf("failed to set device\n");
        return 1;
    }

    size_t number_of_workers = 2;
    blade_ata_bh_initialize(number_of_workers);

    void** input_buffers = (void**)malloc(number_of_workers * sizeof(void*));
    void** output_buffers = (void**)malloc(number_of_workers * sizeof(void*));

    for (int i = 0; i < number_of_workers; i++) {
        size_t input_byte_size = blade_ata_bh_get_input_size() * sizeof(int8_t) * 2;
        input_buffers[i] = (void*)malloc(input_byte_size);
        blade_pin_memory(input_buffers[i], input_byte_size);

        size_t output_byte_size = blade_ata_bh_get_output_size() * sizeof(float);
        output_buffers[i] = (void*)malloc(output_byte_size);
        blade_pin_memory(output_buffers[i], output_byte_size);
    }

    int h = 0, y = 0, i = 0;

    clock_t begin = clock();

    while (i < 16) {
        if (blade_ata_bh_enqueue_b(input_buffers[h], h)) {
            h = (h + 1) % number_of_workers;
        }

        size_t b_id;
        if (blade_ata_bh_dequeue_b(&b_id)) {
            printf("Task B %zu finished.\n", b_id);

            if (blade_ata_bh_enqueue_h(b_id, output_buffers[y], y)) {
                y = (y + 1) % number_of_workers;
            }
        }

        size_t h_id;
        if (blade_ata_bh_dequeue_h(&h_id)) {
            printf("Task H %zu finished.\n", h_id);
            i++;
        }
    }

    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Example finished in %lf s.\n", time_spent);
    double iteration_time = (time_spent / (BLADE_ATA_MODE_BH_ACCUMULATE_RATE * 8)) * 1000;
    printf("Average execution per-iteration: %lf ms.\n", iteration_time);

    blade_ata_bh_terminate();

    for (int i = 0; i < number_of_workers; i++) {
        free(input_buffers[i]);
        free(output_buffers[i]);
    }

    free(input_buffers);
    free(output_buffers);
}
