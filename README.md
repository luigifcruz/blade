## BLADE - Breakthrough Listen Accelerated DSP Engine

This library is meant to be used by the Allen Telescope Array beamformer.

## Dependencies

- GCC >=10.3
- spdlog >=1.5
- CUDA >=11
- Meson >=0.58
- Ninja >=1.10.2
- Cmake >=3.16.3

## Python Dependencies (Optional)

- Pybind11 >=2.4
- Python >=3.6
- Fmt >=6.1

## Installation

Any up-to-date Linux:
```
$ git clone https://github.com/luigifcruz/blade.git
$ cd blade
$ meson build
```

Ubuntu 18.04:
```
$ git clone https://github.com/luigifcruz/blade.git
$ cd blade
$ CC=gcc-10 CXX=g++-10 meson build
```

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

### Jan 6, 2022
Machine: `seti-node8`
Commit: `e84799b688b7358f06534814a25e7b9e5afe96a9`

| -N | -m | -d |  Timestep (ms)  |
|:--:|:--:|:--:|:---------------:|
|  0 |  0 |  0 |       13.5      |
|  0 |  0 |  1 |       21.6      |
|  0 |  1 |  0 |       21.3      |
|  0 |  1 |  1 |      *13.2*     |
|  1 |  0 |  0 |       13.7      |
|  1 |  0 |  1 |       21.4      |
|  1 |  1 |  0 |       21.5      |
|  1 |  1 |  1 |      *13.2*     |

## Implementation Notes

- This code is following [Google C++ Code Style Guide](https://google.github.io/styleguide/cppguide.html).
    - The default line length is 88.
    - Using namespaces is allowed in tests.
- The CUDA C++ Standard Library is being ignored for now because of performance inconsistencies.
- The library is implemented using bleeding-edge features like CUDA 11.4 and C++20.
