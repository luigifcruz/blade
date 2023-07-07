# BLADE - Breakthrough Listen Accelerated DSP Engine

```
                       .-.
        .-""`""-.    |(@ @)
     _/`oOoOoOoOo`\_ \ \-/
    '.-=-=-=-=-=-=-.' \/ \
      `-=.=-.-=.=-'    \ /\
         ^  ^  ^       _H_ \ art by jgs
```

The Blade library provides accelerated signal processing modules for radio telescopes like the Allen Telescope Array. The core library is written in modern C++20 and makes use of just-in-time (JIT) compilation of CUDA kernels to deliver accelerated processing with runtime customizability. Performant Python bindings are also available.

Blade is organized into three parts: Modules, Pipelines, and Runners. The Module performs the data manipulation, the Pipeline integrates multiple Modules together, and the Runner runs multiple Pipelines instances asynchronously to yield the best parallelization.

### Dependencies

- fmtlib >=6.1
- GCC >=10.3
- CUDA >=11
- Meson >=0.58
- Ninja >=1.10.2
- Cmake >=3.16.3

### Python Dependencies (Optional)

- Pybind11 >=2.4
- Python >=3.6

### Test Dependencies (Optional)

- Python Astropy
- Python Pandas

### Documentation Dependencies

```
python -m pip install breathe sphinx_rtd_theme
```

### Installation

Any up-to-date Linux:

```
$ git clone <https://github.com/luigifcruz/blade.git>
$ cd blade
$ meson build
```

Ubuntu 18.04:

```
$ git clone <https://github.com/luigifcruz/blade.git>
$ cd blade
$ CC=gcc-10 CXX=g++-10 meson build
```

### Rules

- All frequencies are in Hertz.
- All angles are in radians.

### Implementation Notes

- This code is following the [Google C++ Code Style Guide](https://google.github.io/styleguide/cppguide.html).
    - The default line length is 88. This can be overridden if necessary. Please, be sensible.
- The CUDA C++ Standard Library is being ignored for now because of performance inconsistencies.
- The library is implemented using bleeding-edge features like CUDA 11.4 and C++20.

# Blade Types

## Foundational

Blade uses standard C++ types listed below. An exception is the half-precision floating-point. In this case, the type is defined with the CUDA headers. This might change in the future to accommodate more platforms/libraries. To reduce code clutter, `typedefs` with the initial letter of each type were created. For example, a 32-bits floating-point becomes `F32`.

```cpp
typedef __half  F16;
typedef float   F32;
typedef double  F64;
typedef int8_t  I8;
typedef int16_t I16;
typedef int32_t I32;
typedef int64_t I64;
```

The complex counterparts of the aforementioned variables are expressed as subtypes of the standard `std::complex` decorator. In this case, a `C` is prepended in the name definition. For example, **complex** 32-bits floating-point becomes `CF32`.

```cpp
typedef std::complex<F16> CF16;
typedef std::complex<F32> CF32;
typedef std::complex<F64> CF64;
typedef std::complex<I8>  CI8;
typedef std::complex<I16> CI16;
typedef std::complex<I32> CI32;
typedef std::complex<I64> CI64;
```

## Structures

Some data patterns that happen often inside the library are standardized to ensure compatibility between modules. These structures are described below.

### Result

This is an enum-like object that defines the status of an operation. For example, a function returning `Result::SUCCESS` means that no errors were encountered. A successful operation can also be interpreted as a zero. In contrast, a failed operation is defined by `Result::ERROR` or other specialization. A failed operation can be interpreted as a positive non-zero value.

```cpp
enum class Result : uint8_t {
    SUCCESS = 0,
    ERROR = 1,
    CUDA_ERROR,
    ASSERTION_ERROR,
};
```

To reduce code clutter, macros to validate the return value of a method are defined.

```cpp
#define BL_CHECK(x) // Prints the Result value and returns.
...

#define BL_CHECK_THROW(x) // Prints the Result value and throw.
...
```

### Vectors

Blade uses a custom class called `Vector` to represent contiguous memory allocation. The vector library supports heterogeneous memory allocation. The locale of the data can be specified with the first template argument being an element of the `Device` enum. Currently, the library only supports `Device::CPU` and `Device::CUDA` arrays. Some devices can be combined with the `operator|`.

```cpp
enum class Device : uint8_t {
    CPU     = 1 << 0,
    CUDA    = 1 << 1,
};

ArrayTensor<Device::CPU, CF32> cpu_array(50); // Allocates CPU array.
ArrayTensor<Device::CUDA, CF32> gpu_array(50); // Allocates CUDA array.

// Allocates managed memory available in the host (CPU) and device(CUDA).
ArrayTensor<Device::CUDA | Device::CPU, CF32> managed_array(50);
```

The custom library offers compile-time checks to ensure the requested data locale of a method matches the array passed as an argument.

```cpp
Result processor(ArrayTensor<Device::CUDA, CF32>& gpu_array) {
...
}

ArrayTensor<Device::CPU, CF32> cpu_array(64);
processor(cpu_array); // Fails! Passing CPU array to method expecting CUDA.

ArrayTensor<Device::CUDA, CF32> gpu_array(64);
processor(gpu_array); // Works!

ArrayTensor<Device::CPU | Device::CUDA, CF32> managed_array(64);
processor(managed_array); // Works! Unified vectors are also supported!
```

Overloads of the `Memory::Copy` method supporting multiple devices provides an easy way to copy elements between arrays.

```cpp
ArrayTensor<Device::CUDA, CF32> gpu_array_src(64);
ArrayTensor<Device::CUDA, CF32> gpu_array_dst(64);
ArrayTensor<Device::CPU, CF32> cpu_array_dst(64);

Memory::Copy(gpu_array_dst, gpu_array_src);

// The same applies for memory copies between devices.
Memory::Copy(cpu_array_dst, gpu_array_src);
```

# Blade Module

A module is the smallest and the most important part of Blade. It is responsible for the actual data manipulation. For example, a Cast module converts an array of elements from a type
to another (e.g. Integer to Float).  A module is not meant to be used alone. It’s supposed to be used inside a Pipeline. This class defines two methods that are automatically called by the Pipeline: 

- **Preprocess**: This is the place to run infrequent data processing. This supports either host or device operations. The code executed here is outside the CUDA Graph. Thus, it’s not meant to be run every iteration. For example, this is the right place to update the phasors of a beamformer module once every several hundreds of milliseconds.
- **Process**: This is the place to run the bulk of operations of the modules. This supports only device operations. All host operations here will be ignored in runtime. This code is executed on every loop inside a CUDA Graph.

The constructor of all modules requires a struct-like object containing the configuration in conjunction with another struct-like object containing the input arrays or buffers required to make the computation. The dynamic array allocations are made at instantiation with no further memory being allocated or deallocated after the processing start. The output arrays and buffers are also generated at instantiation and are available to be consumed with specialized methods like `getOutput` or similar. All modules must implement a `getConfig` method that returns the configuration struct.

```cpp
template<typename IT, typename OT>
class BLADE_API Cast : public Module {
 public:
    struct Config {
        std::size_t inputSize;
    };

    struct Input {
        const ArrayTensor<Device::CUDA, IT>& buf;
    };

    struct Output {
        ArrayTensor<Device::CUDA, OT> buf;
    };

    explicit Cast(const Config& config, const Input& input);

    constexpr ArrayTensor<Device::CUDA, IT>& getInput() {
        return const_cast<ArrayTensor<Device::CUDA, IT>&>(this->input.buf);
    }

    constexpr const ArrayTensor<Device::CUDA, OT>& getOutput() const {
        return this->output.buf;
    }
...
};
```

## Cast

The `Blade::Modules::Cast` module will convert each element of an input array into a desired type and store the result in an output array. The configuration struct of this module expects the `inputSize` with the number of elements of the input array, and `blockSize` with the number of CUDA blocks to use for this operation. The output can be yielded using the `getOutput` method.

## Channelizer

The `Blade::Modules::Channelizer` module will decimate the time-samples of the input data into channels. This is done in the background using an FFT. This function currently only supports a decimation rate of four, but a more flexible implementation is expected. The configuration struct of this module expects the `dims` containing the array data dimensions with the format of the input data, the `fftSize` containing the decimation rate, and the `blockSize` with the number of CUDA blocks to use for this operation. The data output can be yielded by `getOutput`. The `getOutputDims` helper method provides the output data dimensions.

## Beamformer

The Beamformer module will apply the phasors to the input data and generate a beamformed output. This module is radio-telescope-dependent. Currently, only Allen Telescope Array is supported with the `Blade::Modules::Beamformer::ATA` class. Support for more telescopes is expected. The configuration struct of this module expects the array data dimensions with the format of the input data, and the `blockSize` with the number of CUDA blocks to use for this operation. The data output can be yielded by the `getOutput` method. A few helper methods like `getInputSize`, `getOutputSize`, and `getPhasorsSize` are also available with useful buffer size data.

# Blade Pipeline

The Pipeline class executes a single or multiple Modules in sequential order. The execution process is accelerated by CUDA Graph automatically.

A module is added to the execution sequence by calling `this->connect(3)` inside the constructor of a class that inherits the `Blade::Pipeline` object. All classes should implement a `run(N)` method with application-specific operations. This method should receive CPU inputs and outputs, register the asynchronous copy of the inputs from the CPU to GPU using the `this->copy(2)` method, register the computation step with `this->compute(0)`, and finally register the asynchronous copy of the outputs from the GPU to CPU.

Note that all device operations (compute, copy, etc) inside a Pipeline instance are executed in an asynchronous fashion on a private stream. The `this->isSynchronized(0)` method can be utilized to check if all Pipeline instance operations are finished. The `this->synchronize(0)` method can be used to manually synchronize operations.

```cpp
template<typename IT, typename OT>
class Test : public Pipeline {
 public:
    explicit Test(const typename Modules::Channelizer<IT, OT>::Config& config) {
        this->connect(channelizer, config, {input});
    }

    constexpr const std::size_t getInputSize() const {
        return channelizer->getBufferSize();
    }

    Result run(const ArrayTensor<Device::CPU, IT>& input,
                     ArrayTensor<Device::CPU, OT>& output) {
        BL_CHECK(this->copy(channelizer->getInput(), input));
        BL_CHECK(this->compute());
        BL_CHECK(this->copy(output, channelizer->getOutput()));

        return Result::SUCCESS;
    }

 private:
    ArrayTensor<Device::CUDA, IT> input;
    std::shared_ptr<Modules::Channelizer<IT, OT>> channelizer;
};
```

# Blade Runner 🦄

The Runner class will create multiple instances of a Pipeline and run them in parallel. This helps increase the parallel GPU resources utilization. This is possible because a pipeline instance can copy data while another instance is processing data. The exact order of operations can vary according to the device and compute capabilities. The asynchronous Pipeline operations will be scheduled automatically by the Nvidia driver.

Since the Runner leverages the asynchronous nature of the Pipeline execution, the submission of new jobs for processing should also be performed in an asynchronous fashion. The simplest way for doing that is a Queue interface. To accomplish this, the `Blade::Runner` object offers two methods: The `enqueue(1)` to insert a new job into an available worker, and the `dequeue(1)` to check if any job has finished processing. To help keep track of which job corresponds to each buffer, a `std::size_t` variable can be returned in the `enqueue(1)` method, and later retrieved in the `dequeue(1)` method.

When the `enqueue` method is called, the first argument should be a lambda function that accepts a reference of a worker as the first argument. This lambda is responsible to submit the input and output buffers to the pipeline worker using the `worker.run(X)` method. The worker reference passed to the lambda points to the first available internal pipeline instance. To learn more about lambda expressions, please refer to the [C++ reference](https://en.cppreference.com/w/cpp/language/lambda). This lambda also has to return a `std::size_t` value that works as an identification for the job being submitted. Note that in the example below, the lambda is capturing the current context using the `[&](...` decorator. This means that all the variables available outside the lambda are accessible by reference.

```cpp
// Create a poll of workers with a Runner.
auto runner = Blade::Runner<Pipeline>::New(numberOfWorkers, pipelineConfig);

// Enqueue a job to the first available worker.
// The lambda should return an integer with a jobId.
// This jobId can be used to keep track of different jobs.
runner->enqueue([&](auto& worker){
    worker.run(jobInputData, jobOutputData);
    return jobId;
});

// Query Runner if any job has finished.
// If a job has finished, this method will return True.
// This means that the input and output data is ready to be used.
// The jobId from the enqueue step will be copied to the first argument.
runner->dequeue(id);
```

# Implementation Notes
## Calling Blade from C
It's possible to call Blade functions from C by writing an interface file. The recommended design pattern is described below. A complete example of a working C interface can be found at `/tests/pipelines/ata`. A C-header file `api.h` is created containing all interfacing methods from C++ and C. This file can't contain any C++ header or types. These methods are then defined inside a C++ source file called `api.cc`. Here, it's recommended to store the runtime variables with a singleton object. This can be achieved by defining a `static struct`. Finally, Blade can be used normally inside a C application `main.c` by calling the methods defined in the header file.
```c
// api.h

void blade_initialize(size_t number_of_workers);
void blade_enqueue(void* input_ptr, void* output_ptr, size_t* id);
void blade_dequeue(size_t* id);
```
```cpp
// api.cc

extern "C" {
#include "api.h"
}

#include <blade/base.h>

using namespace Blade;

static std::unique_ptr<Runner<ModeB>> runner;

void blade_initialize(size_t number_of_workers) {
...
}

void blade_enqueue(void* input_ptr, void* output_ptr, size_t* id) {
...
}

void blade_dequeue(size_t* id) {
...
}
```
```c
// main.c

#include "mode_b.h"

int main(int argc, char **argv) {
size_t number_of_workers = 2;
blade_ata_b_initialize(number_of_workers);

...

for (int i = 0; i < 510; i++) {
    if (blade_ata_b_enqueue(input_buffers[h], output_buffers[h], i)) {
        h = (h + 1) % number_of_workers;
    }

    size_t id;
    if (blade_ata_b_dequeue(&id)) {
        printf("Task %zu finished.\n", id);
    }
}

...

}
```
