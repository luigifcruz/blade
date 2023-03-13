import time
from random import random
import blade as bl
import numpy as np

class Test(bl.Pipeline):
    detector: bl.Detector

    def __init__(self, input_dims, detector_config: bl.Detector.Config):
        bl.Pipeline.__init__(self)
        self.input = bl.vector.cuda.cf32.ArrayTensor(input_dims)
        _config = detector_config
        _input = bl.Detector.Input(self.input)
        self.detector = self.connect(_config, _input)

    def run(self, input: bl.vector.cpu.cf32,
                  output: bl.vector.cpu.f32):
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

    input_dims = bl.vector.ArrayShape(NBEAMS, NCHANS, NTIME, NPOLS)
    output_dims = bl.vector.ArrayShape(NBEAMS, NCHANS, NTIME, OUTPOLS)

    #
    # Blade Implementation
    #

    detector_config = bl.Detector.Config(
        integration_size = TFACT,
        number_of_output_polarizations = OUTPOLS,
        
        block_size = 512
    )

    mod = Test(input_dims, detector_config)

    bl_input_raw = bl.vector.cpu.cf32.ArrayTensor(input_dims)
    bl_output_raw = bl.vector.cpu.f32.ArrayTensor(output_dims)

    bl_input = np.array(bl_input_raw, copy=False).reshape(input_dims.shape)
    bl_output = np.array(bl_output_raw, copy=False).reshape(output_dims.shape)

    np.copyto(bl_input, [[[[0.21471034+0.14777748j, 0.19602516+0.33223143j]], [[0.9428768 +0.47694337j, 0.87185407+0.7431545j ]]]])

    start = time.time()
    for _ in range(10):
        mod.run(bl_input_raw, bl_output_raw)
    print(f"Detection with Blade took {time.time()-start:.2f} s.")

    print(bl_input)
    print(bl_output)

    assert np.allclose(bl_output, [[[[2.167423]], [[24.288998]]]])
    print("Test successfully completed!")
