## BLADE - Breakthrough Listen Accelerated DSP Engine

This library is meant to be used by the Allen Telescope Array beamformer.

## Dependencies

- GCC >=10.3
- PyBind11 >=2.7.8
- Python >=3.9
- spdlog >=1.8
- CUDA >=11.4

## Pipeline Benchmark   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
   202294  325.296    0.002 1411.103    0.007 correlate.py:16(do_npoint_spec)
414296567  288.778    0.000  288.778    0.000 {built-in method numpy.fft._pocketfft_internal.execute}
414296568  195.792    0.000  792.828    0.000 _pocketfft.py:122(fft)
414296567  185.657    0.000  521.687    0.000 _pocketfft.py:49(_raw_fft)
414296568  142.099    0.000 1082.777    0.000 <__array_function__ internals>:2(fft)
415712620/415308032  114.589    0.000  908.557    0.000 {built-in method numpy.core._multiarray_umath.implement_array_function}
414296567   47.252    0.000   47.252    0.000 {built-in method numpy.core._multiarray_umath.normalize_axis_index}
414296568   40.362    0.000   40.362    0.000 _pocketfft.py:78(_get_forward_norm)
414296568   37.841    0.000   37.841    0.000 _pocketfft.py:118(_fft_dispatcher)
414296568   34.987    0.000   34.987    0.000 {built-in method numpy.asarray}
       14    1.708    0.122    1.708    0.122 {method 'astype' of 'numpy.ndarray' objects}
        1    1.326    1.326 1419.759 1419.759 correlate.py:1(<module>)
       14    1.155    0.083    1.155    0.083 {built-in method numpy.fromfile}
       14    0.939    0.067    3.806    0.272 guppi.py:138(read_next_block)
   202294    0.385    0.000    2.592    0.000 numeric.py:76(zeros_like)
   809168    0.314    0.000    3.418    0.000 <__array_function__ internals>:2(vdot)
   202297    0.312    0.000    0.312    0.000 {built-in method numpy.zeros}
   202294    0.227    0.000    0.634    0.000 <__array_function__ internals>:2(empty_like)
   202308    0.171    0.000    0.171    0.000 {method 'reshape' of 'numpy.ndarray' objects}
   202295    0.151    0.000    1.260    0.000 <__array_function__ internals>:2(copyto)
   202294    0.135    0.000    2.859    0.000 <__array_function__ internals>:2(zeros_like)
### Nov 1, 2021
Machine: `seti-node8`
Commit: `9b1a3569ea861d4bf0bdaf7a3b72706ce537ab99`

| -N | -m | -d |  Timestep (ms)  |
|:--:|:--:|:--:|:---------------:|
|  0 |  0 |  0 |       13.6      |
|  0 |  0 |  1 |       19.3      |
|  0 |  1 |  0 |       19.0      |
|  0 |  1 |  1 |      *12.7*     |
|  1 |  0 |  0 |       13.5      |
|  1 |  0 |  1 |       19.6      |
|  1 |  1 |  0 |       19.0      |
|  1 |  1 |  1 |      *12.7*     |

## Implementation Notes

- This code is following [Google C++ Code Style Guide](https://google.github.io/styleguide/cppguide.html).
    - The default line length is 88.
    - Using namespaces is allowed in tests.
- The CUDA C++ Standard Library is being ignored for now because of performance inconsistencies.
- The library is implemented using bleeding-edge features like CUDA 11.4 and C++20.
