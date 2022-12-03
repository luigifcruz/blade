import time
import blade as bl
import numpy as np
from random import random

class Test(bl.Pipeline):
    polarizer: bl.Polarizer

    def __init__(self, input_dims, config: bl.Polarizer.Config):
        bl.Pipeline.__init__(self)
        self.input = bl.vector.cuda.cf32.ArrayTensor(input_dims)
        _config = config
        _input = bl.Polarizer.Input(self.input)
        self.polarizer = self.connect(_config, _input)

    def run(self, input: bl.vector.cpu.cf32,
                  output: bl.vector.cpu.cf32):
        self.copy(self.polarizer.input(), input)
        self.compute()
        self.copy(output, self.polarizer.output())
        self.synchronize()


if __name__ == "__main__":
    NBEAMS = 2
    NCHANS = 192
    NTIME = 8750
    NPOLS = 2

    input_dims = bl.vector.ArrayDimensions(NBEAMS, NCHANS, NTIME, NPOLS)
    output_dims = bl.vector.ArrayDimensions(NBEAMS, NCHANS, NTIME, NPOLS)

    #
    # Blade Implementation
    #

    config = bl.Polarizer.Config(
        mode = bl.Polarizer.Mode.XY2LR,
        block_size = 512
    )

    mod = Test(input_dims, config)

    bl_input_raw = bl.vector.cpu.cf32.ArrayTensor(input_dims)
    bl_output_raw = bl.vector.cpu.cf32.ArrayTensor(output_dims)

    bl_input = np.array(bl_input_raw, copy=False).reshape(input_dims.shape)
    bl_output = np.array(bl_output_raw, copy=False).reshape(output_dims.shape)

    np.copyto(bl_input, np.random.random(size=bl_input.shape) + 1j*np.random.random(size=bl_input.shape))

    start = time.time()
    mod.run(bl_input_raw, bl_output_raw)
    print(f"Detection with Blade took {time.time()-start:.2f} s.")

    #
    # Python Implementation
    #

    py_input = bl_input.flatten().view(np.float32)
    py_output = np.zeros((NBEAMS * NCHANS * NTIME * NPOLS), dtype=np.complex64).view(np.float32)

    start = time.time()
    py_output[0::2] = py_input[0::2] + py_input[1::2]
    py_output[1::2] = py_input[0::2] - py_input[1::2]
    print(f"Detection with Python took {time.time()-start:.2f} s.")

    py_output = py_output.view(np.complex64)
    py_output = py_output.reshape((NBEAMS, NCHANS, NTIME, NPOLS))

    #
    # Compare Results
    #

    assert np.allclose(bl_output, py_output, rtol=0.1)
    print("Test successfully completed!")
