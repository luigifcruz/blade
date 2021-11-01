import numpy as np


class Generic:

    def __init__(self, NANTS: int, NCHANS: int, NTIME: int, NPOLS: int, NFFT: int):
        self.NANTS = NANTS
        self.NCHANS = NCHANS
        self.NTIME = NTIME
        self.NPOLS = NPOLS
        self.NFFT = NFFT

        self.input_len = (self.NANTS * self.NCHANS * self.NTIME * self.NPOLS)
        self.input_dims = (self.NANTS, self.NCHANS, self.NTIME, self.NPOLS)
        self.output_dims = (self.NANTS, self.NCHANS*4, self.NTIME//4, self.NPOLS)

    def process(self):
        _a = np.random.uniform(-int(2**8/2), int(2**8/2), self.input_len)
        _b = np.random.uniform(-int(2**8/2), int(2**8/2), self.input_len)
        _buffer = np.round(_a + _b * 1j).astype(np.complex64)
        _buffer = _buffer.reshape(self.NANTS * self.NCHANS * self.NTIME, self.NPOLS)

        self.input = np.copy(_buffer)

        for i in range(0, _buffer.shape[0], self.NFFT):
            _a = _buffer[i:i+self.NFFT]
            for pol in range(self.NPOLS):
                _a[:, pol] = np.fft.fft(_a[:, pol])

        self.output = _buffer

    def getInputData(self):
        return self.input

    def getOutputData(self):
        return self.output
