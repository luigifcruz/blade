import sys
import time
import blade as bl
import numpy as np

class Test(bl.Pipeline):
    channelizer: bl.Channelizer

    def __init__(self, input_shape, rate):
        bl.Pipeline.__init__(self)
        self.input = bl.cuda.cf32.ArrayTensor(input_shape)
        _config = bl.Channelizer.Config(rate, 512)
        _input = bl.Channelizer.Input(self.input)
        self.channelizer = self.connect(_config, _input)

    def run(self, input: bl.cpu.cf32.ArrayTensor,
                  output: bl.cpu.cf32.ArrayTensor):
        self.copy(self.channelizer.input(), input)
        self.compute()
        self.copy(output, self.channelizer.output())
        self.synchronize()

def trial(A, F, T, P, C):
    input_shape = (A, F, T, P)
    output_shape = (A, F * C, T // C, P)

    # Initialize Blade pipeline.
    mod = Test(input_shape, C)

    # Generate test data with Python.
    _a = np.random.uniform(-int(2**16/2), int(2**16/2), input_shape)
    _b = np.random.uniform(-int(2**16/2), int(2**16/2), input_shape)
    input = np.array(_a + _b * 1j).astype(np.complex64)

    # Compute the FFT sizes.
    nspecs = T // C

    # Define output buffer.
    _a = np.random.uniform(-int(2**16/2), int(2**16/2), output_shape)
    _b = np.random.uniform(-int(2**16/2), int(2**16/2), output_shape)
    output = np.array(_a + _b * 1j).astype(np.complex64)

    # Import test data from Python to Blade.
    bl_input = bl.cpu.cf32.ArrayTensor(input_shape)
    bl_output = bl.cpu.cf32.ArrayTensor(output_shape)

    np.copyto(bl_input.asnumpy(), input)
    np.copyto(bl_output.asnumpy(), output)

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
    assert np.allclose(bl_output.asnumpy(), output, rtol=0.01)
    print("Test successfully completed!")


if __name__ == "__main__":
    trial(int(sys.argv[1]),
          int(sys.argv[2]),
          int(sys.argv[3]), 
          int(sys.argv[4]), 
          int(sys.argv[5]))
