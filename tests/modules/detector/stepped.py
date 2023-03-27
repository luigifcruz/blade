import time
from random import random
import blade as bl
import numpy as np

class Test(bl.Pipeline):
    detector: bl.Detector

    def __init__(self, input_shape, detector_config: bl.Detector.Config):
        bl.Pipeline.__init__(self)
        self.input = bl.cuda.cf32.ArrayTensor(input_shape)
        _config = detector_config
        _input = bl.Detector.Input(self.input)
        self.detector = self.connect(_config, _input)

    def run(self, input: bl.cpu.cf32,
                  output: bl.cpu.f32):
        self.copy(self.detector.input(), input)
        self.compute()
        self.copy(output, self.detector.output())
        self.synchronize()


if __name__ == "__main__":
    NTIME = 1
    TFACT = 10
    NCHANS = 2
    OUTPOLS = 1
    NBEAMS = 1
    NPOLS = 2

    input_shape = (NBEAMS, NCHANS, NTIME, NPOLS)
    output_shape = (NBEAMS, NCHANS, NTIME, OUTPOLS)

    #
    # Blade Implementation
    #

    detector_config = bl.Detector.Config(
        integration_size = TFACT,
        number_of_output_polarizations = OUTPOLS,
        
        block_size = 512
    )

    mod = Test(input_shape, detector_config)

    bl_input_raw = bl.cpu.cf32.ArrayTensor(input_shape)
    bl_output_raw = bl.cpu.f32.ArrayTensor(output_shape)

    bl_input = bl_input_raw.asnumpy()
    bl_output = bl_output_raw.asnumpy()

    np.copyto(bl_input, [[[[0.21471034+0.14777748j, 0.19602516+0.33223143j]], [[0.9428768 +0.47694337j, 0.87185407+0.7431545j ]]]])

    start = time.time()
    for _ in range(10):
        mod.run(bl_input_raw, bl_output_raw)
    print(f"Detection with Blade took {time.time()-start:.2f} s.")

    print(bl_input)
    print(bl_output)

    assert np.allclose(bl_output, [[[[2.167423]], [[24.288998]]]])
    print("Test successfully completed!")
