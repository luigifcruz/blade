import time
import blade as bl
import numpy as np

class Test(bl.Pipeline):
    channelizer: bl.Channelizer
    input = bl.vector.cuda.cf32()

    def __init__(self, dims: bl.ArrayDims, fft_size):
        bl.Pipeline.__init__(self)
        _config = bl.Channelizer.Config(dims, fft_size, 512)
        _input = bl.Channelizer.Input(self.input)
        self.channelizer = self.connect(_config, _input)

    def bufferSize(self):
        return self.channelizer.bufferSize()

    def run(self, input: bl.vector.cpu.cf32,
                  output: bl.vector.cpu.cf32):
        self.copy(self.channelizer.input(), input)
        self.compute()
        self.copy(output, self.channelizer.output())
        self.synchronize()


if __name__ == "__main__":
    # Specify dimension of array.
    NFFT = 4
    d = bl.ArrayDims(NBEAMS=1, NANTS=20, NCHANS=96, NTIME=35000, NPOLS=2)

    # Initialize Blade pipeline.
    mod = Test(d, NFFT)

    # Generate test data with Python.
    _a = np.random.uniform(-int(2**16/2), int(2**16/2), mod.bufferSize())
    _b = np.random.uniform(-int(2**16/2), int(2**16/2), mod.bufferSize())
    _c = np.array(_a + _b * 1j).astype(np.complex64)
    input = _c.reshape((d.NANTS * d.NCHANS * d.NTIME, d.NPOLS))

    output = np.zeros_like(input, dtype=np.complex64)

    # Import test data from Python to Blade.
    bl_input = bl.vector.cpu.cf32(mod.bufferSize())
    bl_output = bl.vector.cpu.cf32(mod.bufferSize())

    np.copyto(np.array(bl_input, copy=False), input.flatten())
    np.copyto(np.array(bl_output, copy=False), output.flatten())

    # Beamform with Blade.
    start = time.time()
    mod.run(bl_input, bl_output)
    print(f"Channelization with Blade took {time.time()-start:.2f} s.")

    # Beamform with Numpy.
    start = time.time()
    for i in range(0, input.shape[0], NFFT):
            _a = input[i:i+NFFT]
            _b = output[i:i+NFFT]
            for pol in range(d.NPOLS):
                _b[:, pol] = np.fft.fft(_a[:, pol])
    print(f"Channelization with Numpy took {time.time()-start:.2f} s.")

    # Check both answers.
    assert np.allclose(np.array(bl_output, copy=False), output.flatten(), rtol=0.01)
    print("Test successfully completed!")
