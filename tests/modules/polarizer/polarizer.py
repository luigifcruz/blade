import time
import blade as bl
import numpy as np
from random import random


@Pipeline
class Test():
    def __init__(self, input_shape, config: bl.Polarizer.Config):
        self.input = bl.cuda.cf32.ArrayTensor(input_shape)

        self.mod.polarizer = self.connect(
            _config,
            bl.Polarizer.Input(self.input)
        )

    @TransferIn
    def transferIn(self, input):
        self.copy(self.polarizer.input(), input)

    @TransferOut
    def transferOut(self, output):
        self.copy(output, self.polarizer.output())

if __name__ == "__main__":
    shape = (2, 192, 8750, 2)

    #
    # Blade Implementation
    #

    config = bl.Polarizer.Config(
        mode = bl.Polarizer.Mode.XY2LR,
        block_size = 512
    )

    mod = Test(shape, config)

    bl_input_raw = bl.cpu.cf32.ArrayTensor(shape)
    bl_output_raw = bl.cpu.cf32.ArrayTensor(shape)

    bl_input = bl_input_raw.asnumpy()
    bl_output = bl_output_raw.asnumpy()
    np.copyto(bl_input, np.random.random(size=shape) + 1j*np.random.random(size=shape))

    start = time.time()
    mod.run(bl_input_raw, bl_output_raw)
    print(f"Detection with Blade took {time.time()-start:.2f} s.")

    #
    # Python Implementation
    #

    py_input = bl_input.flatten().view(np.complex64)
    py_output = np.zeros(len(bl_input_raw.shape()), dtype=np.complex64)

    start = time.time()
    py_output[0::2] = py_input[0::2] + 1j * py_input[1::2]
    py_output[1::2] = py_input[0::2] - 1j * py_input[1::2]
    print(f"Detection with Python took {time.time()-start:.2f} s.")

    py_output = py_output.view(np.complex64)
    py_output = py_output.reshape(shape)

    #
    # Compare Results
    #

    assert np.allclose(bl_output, py_output, rtol=0.1)
    print("Test successfully completed!")
