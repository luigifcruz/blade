# Import the blade library
import blade as bl

# Define the synchronous pipeline class
@bl.runner
class Pipeline:
    # Initialize the pipeline with shape and configuration
    def __init__(self, shape, config):
        # Create input and output buffers with the specified shape
        self.input.buf = bl.array_tensor(shape, dtype=bl.cf32)
        self.output.buf = bl.array_tensor(shape, dtype=bl.cf32)

        # Initialize the polarizer module with the given configuration and input buffer
        self.module.polarizer = bl.module(bl.polarizer, config, self.input.buf)

    # Transfer data from the provided buffer to the input buffer
    def transfer_in(self, buf):
        self.copy(self.input.buf, buf)

    # Transfer data from the polarizer's output to the output buffer and then to the provided buffer
    def transfer_out(self, buf):
        self.copy(self.output.buf, self.module.polarizer.get_output())
        self.copy(buf, self.output.buf)

# Define the shape and configuration for the pipeline
shape = (2, 192, 8750, 2)
config = {
    'inputPolarization': bl.pol.xy,
    'outputPolarization': bl.pol.lr,
}

# Create an instance of the pipeline
pipeline = Pipeline(shape, config)

# Create host input and output buffers with the specified shape and device
host_input = bl.array_tensor(shape, dtype=bl.cf32, device=bl.cpu)
host_output = bl.array_tensor(shape, dtype=bl.cf32, device=bl.cpu)

# Execute the pipeline synchronously with the host input and output buffers
pipeline(host_input, host_output)