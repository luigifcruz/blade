#ifndef MODE_B_STUB_H
#define MODE_B_STUB_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <cuda_runtime.h>

#include "mode_b.h"

int mode_b_init();
int mode_b_setup();
int mode_b_loop(int iterations);
int mode_b_terminate();

#endif
