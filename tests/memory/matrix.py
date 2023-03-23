import math
import blade as bl
import numpy as np
from numpy import random

# Human Version

vec_bl = bl.cpu.f32.ArrayTensor((1, 1, 10, 2))
vec_np = vec_bl.asnumpy()

for i in range(10):
    vec_np[0, 0, i, 0] = 0.69
    vec_np[0, 0, i, 1] = 0.42

print(vec_np)
print(vec_bl.asnumpy())

assert np.allclose(vec_np, vec_bl.asnumpy(), rtol=0.1)

for i in range(10):
    print(vec_bl[0, 0, i, 0], vec_bl[0, 0, i, 1])

    assert math.isclose(vec_bl[0, 0, i, 0], 0.69, rel_tol=1e-5)
    assert math.isclose(vec_bl[0, 0, i, 1], 0.42, rel_tol=1e-5)

# Super Human Version

vec_bl = bl.cpu.f32.ArrayTensor((3, 13, 9, 11))
vec_np = vec_bl.asnumpy()

vec_np[:, :, :, :] = random.rand(*vec_np.shape)

assert np.allclose(vec_np, vec_bl.asnumpy(), rtol=0.1)

for a in range(vec_np.shape[0]):
    for b in range(vec_np.shape[1]):
        for c in range(vec_np.shape[2]):
            for d in range(vec_np.shape[3]):
                assert math.isclose(
                    vec_bl[a, b, c, d], 
                    vec_np[a, b, c, d], rel_tol=1e-5)

print("All tests successful!")