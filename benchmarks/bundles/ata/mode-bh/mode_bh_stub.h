#ifndef MODE_BH_STUB_H
#define MODE_BH_STUB_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <cuda_runtime.h>

#include "mode_bh.h"

int mode_bh_init();
int mode_bh_setup();
int mode_bh_loop(int iterations);
int mode_bh_terminate();

#endif
