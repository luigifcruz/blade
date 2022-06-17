#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#include "mode_b.h"

int main(int argc, char **argv) {
    if (argc == 3 && !blade_use_device(atoi(argv[2]))) {
        printf("failed to set device\n");
        return 1;
    }

    size_t number_of_workers = 1;
    blade_vla_b_initialize(number_of_workers);

    void** input_buffers = (void**)malloc(number_of_workers * sizeof(void*));
    void** phasors_buffers = (void**)malloc(number_of_workers * sizeof(void*));
    void** output_buffers = (void**)malloc(number_of_workers * sizeof(void*));

    for (int i = 0; i < number_of_workers; i++) {
        size_t input_byte_size = blade_vla_b_get_input_size() * sizeof(int8_t) * 2;
        input_buffers[i] = (void*)malloc(input_byte_size);
        blade_pin_memory(input_buffers[i], input_byte_size);

        size_t phasors_byte_size = blade_vla_b_get_phasors_size() * sizeof(float) * 2;
        phasors_buffers[i] = (void*)malloc(phasors_byte_size);
        blade_pin_memory(phasors_buffers[i], phasors_byte_size);

        size_t output_byte_size = blade_vla_b_get_output_size() * BLADE_VLA_MODE_B_OUTPUT_NCOMPLEX_BYTES;
        output_buffers[i] = (void*)malloc(output_byte_size);
        blade_pin_memory(output_buffers[i], output_byte_size);
    }

    int h = 0, i = 0;

    clock_t begin = clock();

    while (i < 16) {
        if (blade_vla_b_enqueue(input_buffers[h], phasors_buffers[h], output_buffers[h], i)) {
            h = (h + 1) % number_of_workers;
        }

        size_t id;
        if (blade_vla_b_dequeue(&id)) {
            printf("Task %zu finished.\n", id);
            i++;
        }
    }

    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Example finished in %lf s.\n", time_spent);
    double iteration_time = (time_spent / 16) * 1000;
    printf("Average execution per-iteration: %lf ms.\n", iteration_time);

    blade_vla_b_terminate();

    for (int i = 0; i < number_of_workers; i++) {
        free(input_buffers[i]);
        free(phasors_buffers[i]);
        free(output_buffers[i]);
    }

    free(input_buffers);
    free(phasors_buffers);
    free(output_buffers);
}
