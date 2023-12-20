import numpy as np
import blade as bl


if __name__ == "__main__":
    shape = (2, 192, 8750, 2)

    host_input = bl.array_tensor(shape, dtype=bl.f32, device=bl.cpu)

    np.copyto(host_input.as_numpy(), np.random.uniform(-int(2**8/2), int(2**8/2), shape).astype(np.float32))

    # CPU -> CPU

    host_output = bl.array_tensor(shape, dtype=bl.f32, device=bl.cpu)
    bl.copy(host_input, host_output)
    assert np.allclose(host_input.as_numpy(), host_output.as_numpy(), rtol=0.1)

    print("Test successfully completed!")
