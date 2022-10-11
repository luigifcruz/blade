import sys
import time
import blade as bl
import numpy as np

class Test(bl.Pipeline):
    channelizer: bl.Channelizer

    def __init__(self, input_dims, rate):
        bl.Pipeline.__init__(self)
        self.input = bl.vector.cuda.cf32.ArrayTensor(input_dims)
        _config = bl.Channelizer.Config(rate, 512)
        _input = bl.Channelizer.Input(self.input)
        self.channelizer = self.connect(_config, _input)

    def run(self, input: bl.vector.cpu.cf32.ArrayTensor,
                  output: bl.vector.cpu.cf32.ArrayTensor):
        self.copy(self.channelizer.input(), input)
        self.compute()
        self.copy(output, self.channelizer.output())
        self.synchronize()

def trial(A, F, T, P, C):
    input_dims = bl.vector.ArrayDimensions(A, F, T, P)
    output_dims = bl.vector.ArrayDimensions(A, F * C, T // C, P)

    # Initialize Blade pipeline.
    mod = Test(input_dims, C)

    # Generate test data with Python.
    _a = np.random.uniform(-int(2**16/2), int(2**16/2), len(input_dims))
    _b = np.random.uniform(-int(2**16/2), int(2**16/2), len(input_dims))
    _c = np.array(_a + _b * 1j).astype(np.complex64)
    input = _c.reshape(input_dims.shape)

    # Compute the FFT sizes.
    nspecs = T // C

    # Define output buffer.
    _a = np.random.uniform(-int(2**16/2), int(2**16/2), len(output_dims))
    _b = np.random.uniform(-int(2**16/2), int(2**16/2), len(output_dims))
    _c = np.array(_a + _b * 1j).astype(np.complex64)
    output = _c.reshape(output_dims.shape)

    # Import test data from Python to Blade.
    bl_input = bl.vector.cpu.cf32.ArrayTensor(input_dims)
    bl_output = bl.vector.cpu.cf32.ArrayTensor(output_dims)

    np.copyto(np.array(bl_input, copy=False), input.flatten())
    np.copyto(np.array(bl_output, copy=False), output.flatten())

    # Channelize with Blade.
    start = time.time()
    mod.run(bl_input, bl_output)
    print(f"Channelize with Blade took {time.time()-start:.2f} s.")

    # Channelize with Numpy.
    start = time.time()
    for ibeam in range(A):
        beam = input[ibeam]

        for ichan in range(F):
            time_pol = beam[ichan]

            for ipol in range(P):
                time_arr = time_pol[:, ipol]

                for ispec in range(nspecs):
                    output[ibeam,
                            ichan*C :
                            (ichan+1)*C,
                            ispec, ipol] =\
                            np.fft.fftshift(np.fft.fft(time_arr[ispec*C:(ispec+1)*C]))
    print(f"Channelize with Numpy took {time.time()-start:.2f} s.")

    # Check both answers.
    assert np.allclose(np.array(bl_output, copy=False), output.flatten(), rtol=0.01)
    print("Test successfully completed!")


if __name__ == "__main__":
    trial(int(sys.argv[1]),
          int(sys.argv[2]),
          int(sys.argv[3]), 
          int(sys.argv[4]), 
          int(sys.argv[5]))
