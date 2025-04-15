import sys
import numpy as np
import blade as bl

@bl.runner
class Pipeline:
    def __init__(self, input_shape, output_shape, config):
        self.input.buf = bl.array_tensor(input_shape, dtype=bl.cf32)
        self.output.buf = bl.array_tensor(output_shape, dtype=bl.cf32)

        self.module.correlator = bl.module(bl.correlator, config, self.input.buf)

    def transfer_in(self, buf):
        self.copy(self.input.buf, buf)

    def transfer_out(self, buf):
        self.copy(self.output.buf, self.module.correlator.get_output())
        self.copy(buf, self.output.buf)


def test(A, F, T, P, I, S, C, B):
    # This assumes that the input data was already transferred to the frequency domain.
    number_of_antennas = A
    number_of_channels = F
    number_of_samples = T
    number_of_polarizations = P
    integration_rate = I

    number_of_baselines = int((number_of_antennas * (number_of_antennas + 1)) / 2)

    input_shape = (number_of_antennas, number_of_channels, number_of_samples, number_of_polarizations)
    output_shape = (number_of_baselines, number_of_channels, 1, 4)

    config = {
        'integration_rate': integration_rate,
        "use_shared_memory": True if S == 1 else False,
        "calculation_mode": bl.calc_mode.integer             if C == 0 else
                            bl.calc_mode.single_precision_fp if C == 1 else
                            bl.calc_mode.double_precision_fp if C == 2 else exit(),
        'block_size': B
    }

    host_input = bl.array_tensor(input_shape, dtype=bl.cf32, device=bl.cpu)
    host_output = bl.array_tensor(output_shape, dtype=bl.cf32, device=bl.cpu)

    bl_input = host_input.as_numpy()
    bl_output = host_output.as_numpy()

    np.copyto(bl_input, np.random.uniform(0, 255, size=input_shape) + 1j * np.random.uniform(0, 255, size=input_shape))

    #
    # Blade Implementation
    #

    pipeline = Pipeline(input_shape, output_shape, config)
    while True:
        if pipeline(host_input, host_output):
            break

    #
    # Python Implementation
    #

    py_output = np.zeros(output_shape, dtype=np.complex128)

    ibline = 0
    for _ in range(integration_rate):
        for iant1 in range(number_of_antennas):
            for iant2 in range(iant1, number_of_antennas):
                ant1 = np.complex128(bl_input[iant1, ...])
                ant2 = np.complex128(bl_input[iant2, ...])

                py_output[ibline, :, 0, 0] += np.sum(ant1[:, :, 0] * np.conj(ant2[:, :, 0]), axis=1)
                py_output[ibline, :, 0, 1] += np.sum(ant1[:, :, 0] * np.conj(ant2[:, :, 1]), axis=1)
                py_output[ibline, :, 0, 2] += np.sum(ant1[:, :, 1] * np.conj(ant2[:, :, 0]), axis=1)
                py_output[ibline, :, 0, 3] += np.sum(ant1[:, :, 1] * np.conj(ant2[:, :, 1]), axis=1)

                ibline += 1
        ibline = 0

    #
    # Compare Results
    #

    print("Top 10 differences:")
    diff = np.abs(np.abs(bl_output) - np.abs(py_output))
    diff = diff.flatten()
    diff.sort()
    print(diff[-10:])
    print("")
    print("Average difference: ", np.mean(diff))
    print("Maximum difference: ", np.max(diff))
    print("Minimum difference: ", np.min(diff))
    print(bl_output[0, 0, 0, 0], py_output[0, 0, 0, 0])

    assert np.allclose(bl_output, py_output, rtol=0.1)

    print("Test successfully completed!")

if __name__ == "__main__":
    test(int(sys.argv[1]),
         int(sys.argv[2]),
         int(sys.argv[3]),
         int(sys.argv[4]),
         int(sys.argv[5]),
         int(sys.argv[6]),
         int(sys.argv[7]),
         int(sys.argv[8]))
