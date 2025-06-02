import numpy as np
import blade as bl

sklim_vals = {
    3 : {
        256 : [0.698159, 1.49597],
        512 : [0.775046, 1.32542],
        1024 : [0.834186, 1.21695]
    },
    4 : {
        256 : [0.613738, 1.784],
        512 : [0.711612, 1.48684],
        1024 : [0.786484, 1.31218],
    },
    5 : {
        32: [0.172532, 7.11963],
        64: [0.224438, 4.75752],
        256 : [0.526881, 2.18694],
        512 : [0.649093, 1.69044],
        1024 : [0.740405, 1.42332],
        2048 : [0.808641, 1.27145]
    }
}

@bl.runner
class Pipeline:
    def __init__(self, input_shape,  output_shape, config):
        self.input.buf = bl.array_tensor(input_shape, dtype=bl.cf32)
        # self.input.phasors = bl.phasor_tensor(phasor_shape, dtype=bl.cf32)
        self.output.buf = bl.array_tensor(output_shape, dtype=bl.cf32)

        input = (self.input.buf)
        self.module.kurtosis = bl.module(bl.kurtosis, config, input)

    def transfer_in(self, buf):
        self.copy(self.input.buf, buf)

    def transfer_out(self, buf):
        self.copy(self.output.buf, self.module.kurtosis.get_output())
        self.copy(buf, self.output.buf)


if __name__ == "__main__":
    # Specify dimension of array.
    input_shape = (28, 192, 8192, 2)
    output_shape = input_shape #(20, 192, 8192, 2)

    block_size = 256

    config = {
        #'enable_incoherent_beam': True,
        #'enable_incoherent_beam_sqrt': True,
    }

    host_input = bl.array_tensor(input_shape, dtype=bl.cf32, device=bl.cpu)
    # host_phasors = bl.phasor_tensor(phasor_shape, dtype=bl.cf32, device=bl.cpu)
    host_output = bl.array_tensor(output_shape, dtype=bl.cf32, device=bl.cpu)

    bl_input = host_input.as_numpy()
    # bl_phasors = host_phasors.as_numpy()
    bl_output = host_output.as_numpy()

    sample_input = 1 * np.random.normal(size=input_shape) + 2j * np.random.normal(size=input_shape)
    
    # sample pollution
    # sample_input[0, 0, 0, 0] = 100 + 100j

    # np.copyto(bl_input, np.random.random(size=input_shape)+1j*np.random.random(size=input_shape))
    np.copyto(bl_input, sample_input)
    # np.copyto(bl_phasors, np.random.random(size=phasor_shape)+1j*np.random.random(size=phasor_shape))

    #
    # Blade Implementation
    #

    pipeline = Pipeline(input_shape, output_shape, config)
    pipeline(host_input,  host_output)




    #
    # Python Implementation
    #

    # convert to cupy array
    full_block_cp = np.array(bl_input)

    # ################################
    # compute the sk_arr
    
    # assuming mean of 0 and std of 1
    # block = block - np.mean(block)
    # block = block / np.std(block)

    chunksize = full_block_cp.shape[2]
    result = None 
    for bstart in range(0, chunksize, block_size):
        # bstart = block_ind
        bend = bstart + block_size
        block_cp = full_block_cp[:, :, bstart:bend, :]
        
        # block_median = np.median(block_cp)
        # print(block_ind, bstart, bend, block_cp.shape)
        nants, nfreqs, nsamples, npols = block_cp.shape
        m = nsamples

        d_dt = block_cp.real * block_cp.real + block_cp.imag * block_cp.imag

        s1 = d_dt.sum(axis = 2, keepdims = True)
        s2 = (d_dt ** 2).sum(axis = 2, keepdims = True)
        sk_arr = ((m + 1.) / (m - 1.)) * ((m * (s2 / (s1**2))) - 1)
        # done computing sk_arr
        # ################################

        sk_mean = 1

        # TODO: remove assumption
        n_stds = 5

        sk_bounds = sklim_vals[n_stds][block_size]


        mask = np.logical_and(sk_arr > sk_bounds[0], sk_arr < sk_bounds[1])
        nzapped = np.sum(1 - mask)
        print("zapped", nzapped, "of", mask.shape)
        # mask = cp.logical_and(sk_arr > sk_bounds[0], sk_arr < sk_bounds[1])
        
        # we are replacing with 100 here since we're in debug mode
        maskedblock = block_cp * mask + ((100 + 100j) * (1 - mask))
        #maskedblock = (block_cp * mask) # + (block_median * (1 - mask))
        if result is not None:
            result = np.concatenate((result, maskedblock), axis = 2)
        else:
            result = maskedblock
    # maskedblock[np.where(maskedblock == 0)] = 0.0 #np.median(block_cp)

    # py_output = cp.asnumpy(maskedblock)
    py_output = np.array(result)

    # if np.sum(py_output != cp.asnumpy(block_cp)) == 0:
    #    print("WARNING: Highly unlikely -- no change due to SK in GPU code")


    #
    # Compare Results
    # Since kurtosis does an in-place modification
    # maybe we want to compare bl_input with py_output?

    assert np.allclose(bl_output[:-1, :, :, :], py_output[:-1, :, :, :], rtol=0.01)
    assert np.allclose(bl_output[-1, :, :, :], py_output[-1, :, :, :], atol=250)

    # assert np.allclose(bl_output[:-1, :, :, :], py_output[:-1, :, :, :], rtol=0.01)
    # assert np.allclose(bl_output[-1, :, :, :], py_output[-1, :, :, :], atol=250)
    


    print("Test successfully completed!")
