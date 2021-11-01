# BLADE - Breakthrough Listen Accelerated DSP Engine

This library is meant to be used by the Allen Telescope Array beamformer.

## Dependencies

- Clang >=11 or GCC >=10.3
- PyBind11 >=2.7.8
- Python >=3.9
- spdlog >=1.8
- CUDA >=11.4

## Implementation Notes

- This code is following [Google C++ Code Style Guide](https://google.github.io/styleguide/cppguide.html).
    - The default line length is 88.
    - Using namespaces is allowed in tests.
- The CUDA C++ Standard Library is being ignored for now because of performance inconsistencies.
