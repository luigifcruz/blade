#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>

#include "module.h"

int main(int argc, char **argv) {
    if (argc == 3 && (cudaSetDevice(atoi(argv[2])) != 0)) {
        printf("failed to set device\n");
        return 1;
    }

    size_t batch_size = 2;
    module_t mod = init(batch_size);

    void** input_buffers = (void**)malloc(batch_size * sizeof(void*));
    void** output_buffers = (void**)malloc(batch_size * sizeof(void*));

    for (int i = 0; i < batch_size; i++) {
        size_t input_byte_size = get_input_size(mod) * sizeof(int8_t) * 2;
        input_buffers[i] = (void*)malloc(input_byte_size);
        pin_memory(mod, input_buffers[i], input_byte_size);

        size_t output_byte_size = get_output_size(mod) * sizeof(int16_t) * 2;
        output_buffers[i] = (void*)malloc(output_byte_size);
        pin_memory(mod, output_buffers[i], output_byte_size);
    }

    for (int i = 0; i < 255; i++) {
        process(mod, input_buffers, output_buffers);
    }

    deinit(mod);

    for (int i = 0; i < batch_size; i++) {
        free(input_buffers[i]);
        free(output_buffers[i]);
    }

    free(input_buffers);
    free(output_buffers);
}
