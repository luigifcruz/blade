import sys
import numpy as np
import blade as bl

@bl.runner
class Pipeline:
    def __init__(self, input_shape, output_shape, config):
        self.input.buf = bl.array_tensor(input_shape, dtype=bl.cf32)
        self.output.buf = bl.array_tensor(output_shape, dtype=bl.cf32)

        self.module.channelizer = bl.module(bl.channelizer, config, self.input.buf)

    def transfer_in(self, buf):
        self.copy(self.input.buf, buf)

    def transfer_out(self, buf):
        self.copy(self.output.buf, self.module.channelizer.get_output())
        self.copy(buf, self.output.buf)


def test(A, F, T, P, C):
    input_shape = (A, F, T, P)
    output_shape = (A, F * C, T // C, P)

    config = {
        "rate": C,
    }

    host_input = bl.array_tensor(input_shape, dtype=bl.cf32, device=bl.cpu)
    host_output = bl.array_tensor(output_shape, dtype=bl.cf32, device=bl.cpu)

    bl_input = host_input.as_numpy()
    bl_output = host_output.as_numpy()

    np.copyto(bl_input, np.random.random(size=input_shape) + 1j*np.random.random(size=input_shape))
    np.copyto(bl_output, np.random.random(size=output_shape) + 1j*np.random.random(size=output_shape))
 
    #
    # Blade Implementation
    #

    pipeline = Pipeline(input_shape, output_shape, config)
    pipeline(host_input, host_output)

    #
    # Python Implementation
    #

    py_output = np.zeros(output_shape, dtype=np.complex64)

    for ibeam in range(A):
        beam = bl_input[ibeam]

        for ichan in range(F):
            time_pol = beam[ichan]

            for ipol in range(P):
                time_arr = time_pol[:, ipol]

                for ispec in range(T // C):
                    py_output[ibeam,
                              ichan*C :
                              (ichan+1)*C,
                              ispec, ipol] =\
                              np.fft.fftshift(np.fft.fft(time_arr[ispec*C:(ispec+1)*C]))

    #
    # Compare Results
    #

    assert np.allclose(bl_output, py_output, rtol=0.01)
    print("Test successfully completed!")


if __name__ == "__main__":
    test(int(sys.argv[1]),
         int(sys.argv[2]),
         int(sys.argv[3]), 
         int(sys.argv[4]), 
         int(sys.argv[5]))