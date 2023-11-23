import numpy as np
import blade as bl

@bl.runner
class Pipeline:
    def __init__(self, shape, config):
        self.input.buf = bl.array_tensor(shape, dtype=bl.cf32)
        self.output.buf = bl.array_tensor(shape, dtype=bl.cf32)

        self.module.polarizer = bl.module(bl.polarizer, config, self.input.buf)

    def transfer_in(self, buf):
        self.copy(self.input.buf, buf)

    def transfer_out(self, buf):
        self.copy(self.output.buf, self.module.polarizer.get_output())
        self.copy(buf, self.output.buf)


if __name__ == "__main__":
    shape = (2, 192, 8750, 2)

    config = {
        'inputPolarization': bl.pol.xy,
        'outputPolarization': bl.pol.lr,
    }

    host_input = bl.array_tensor(shape, dtype=bl.cf32, device=bl.cpu)
    host_output = bl.array_tensor(shape, dtype=bl.cf32, device=bl.cpu)

    bl_input = host_input.as_numpy()
    bl_output = host_output.as_numpy()

    np.copyto(bl_input, np.random.random(size=shape) + 1j*np.random.random(size=shape))

    #
    # Blade Implementation
    #

    pipeline = Pipeline(shape, config)
    pipeline(host_input, host_output)

    #
    # Python Implementation
    #

    py_input = bl_input.flatten().view(np.complex64)
    py_output = np.zeros(len(host_input.shape), dtype=np.complex64)

    py_output[0::2] = py_input[0::2] + 1j * py_input[1::2]
    py_output[1::2] = py_input[0::2] - 1j * py_input[1::2]

    py_output = py_output.view(np.complex64)
    py_output = py_output.reshape(shape)

    #
    # Compare Results
    #

    assert np.allclose(bl_output, py_output, rtol=0.1)
    print("Test successfully completed!")
