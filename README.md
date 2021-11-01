## BLADE - Breakthrough Listen Accelerated DSP Engine

This library is meant to be used by the Allen Telescope Array beamformer.

## Dependencies

- Clang >=11 or GCC >=10.3
- PyBind11 >=2.7.8
- Python >=3.9
- spdlog >=1.8
- CUDA >=11.4

## Pipeline Benchmark
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
