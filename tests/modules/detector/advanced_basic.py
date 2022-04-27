import time
import blade as bl
import numpy as np

class Test(bl.Pipeline):
    detector: bl.Detector
    input = bl.vector.cuda.cf32()

    def __init__(self, detector_config: bl.Detector.Config):
        bl.Pipeline.__init__(self)
        _config = detector_config
        _input = bl.Detector.Input(self.input)
        self.detector = self.connect(_config, _input)

    def input_size(self):
        return self.detector.input_size()

    def output_size(self):
        return self.detector.output_size()

    def run(self, input: bl.vector.cpu.cf32,
                  output: bl.vector.cpu.f32):
        self.copy(self.detector.input(), input)
        self.compute()
        self.copy(output, self.detector.output())
        self.synchronize()


if __name__ == "__main__":
    #
    # Blade Implementation
    #

    detector_config = bl.Detector.Config(
        number_of_beams = 1,
        number_of_frequency_channels = 3,
        number_of_time_samples = 2,
        number_of_polarizations = 2,

        integration_size = 2,
        number_of_output_polarizations = 4,
        
        block_size = 512
    )

    mod = Test(detector_config)

    bl_input_raw = bl.vector.cpu.cf32(mod.input_size())
    bl_output_raw = bl.vector.cpu.f32(mod.output_size())

    bl_input = np.array(bl_input_raw, copy=False)
    bl_output = np.array(bl_output_raw, copy=False)

    for i in range(len(bl_input)):
        bl_input[i] = 1.0

    print(bl_input)

    mod.run(bl_input_raw, bl_output_raw)

    print(bl_output)

    #
    # Python Implementation
    #

    NTIME = 2
    TFACT = 2
    NCHANS = 3
    OUTPOLS = 4
    NBEAMS = 1
    NPOLS = 2

    out_nsamp=NTIME//TFACT

    output_data = np.zeros(shape=(NBEAMS, NCHANS, NTIME, NPOLS), dtype=np.complex64)
    output_data_detected = np.zeros(shape=(NBEAMS, NCHANS, out_nsamp, OUTPOLS), dtype=np.float32)

    for i in range(len(output_data)):
        output_data[i] = 1.0

    print(output_data)
    
    for ibeam in range(NBEAMS):
        for ichan in range(NCHANS):
            for isamp in range(out_nsamp):
                x = output_data[ibeam, ichan, isamp*TFACT:isamp*TFACT+TFACT, 0] #just to make code more visible
                y = output_data[ibeam, ichan, isamp*TFACT:isamp*TFACT+TFACT, 1] #just to make code more visible

                #x and y have shapes = [TFACT]

                # the below is complex dot product (complex conjugate the second term), so can be expanded
                auto_x = x*np.conj(x) # or: x.real*x.real + x.imag*x.imag... this will definitely be a real value, no .imag part
                auto_y = y*np.conj(y) # or: y.real*y.real + y.imag*y.imag... this will definitely be a real value, no .imag part
                cross  = x*np.conj(y) # or:
                #                           cross_real =  x.real * x.real + y.real*y.real
                #                           cross_imag = -x.real * y.imag + y.real*x.imag

                output_data_detected[ibeam, ichan, isamp, 0] = np.sum(auto_x.real) #real is actually abs() too, because x*xT
                output_data_detected[ibeam, ichan, isamp, 1] = np.sum(auto_y.real) #real is actually abs() too, because y*yT
                output_data_detected[ibeam, ichan, isamp, 2] = np.sum(cross.real)
                output_data_detected[ibeam, ichan, isamp, 3] = np.sum(cross.imag)

    print(output_data_detected)