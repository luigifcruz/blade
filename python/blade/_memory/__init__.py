import blade._internal as bl
import blade._mem_impl as _mem

def _create_array(type, shape, dtype, device):
    _pipeline = bl._Fetch()
    _type = type + '_duet' if _pipeline else type
    _shape = shape
    _dtype = dtype.value
    _device = device.value if device != bl.unified else bl.cuda.value
    _unified = False if device != bl.unified else True

    _caller = _mem

    _caller = getattr(_caller, _device)
    _caller = getattr(_caller, _dtype)
    _caller = getattr(_caller, _type)

    return _caller(_shape, unified=_unified)

def array_tensor(shape, dtype=bl.f32, device=bl.cuda):
    """
    Create a tensor with the specified shape, data type, and device.

    Args:
        shape (tuple): The shape of the tensor.
        dtype (bl.constant, optional): The data type of the tensor (`bl.f32`, `bl.i8`, etc). Defaults to `bl.f32`.
        device (bl.constant, optional): The device to create the tensor on (`bl.cuda`, `bl.cpu`, or `bl.unified`). Defaults to `bl.cuda`.

    Returns:
        Tensor: The created tensor.
    """
    return _create_array("array_tensor", shape, dtype, device)

def phasor_tensor(shape, dtype=bl.f32, device=bl.cuda):
    """
    Creates a phasor tensor with the specified shape, data type, and device.

    Args:
        shape (tuple): The shape of the tensor.
        dtype (bl.constant, optional): The data type of the tensor (`bl.f32`, `bl.i8`, etc). Defaults to `bl.f32`.
        device (bl.constant, optional): The device to create the tensor on (`bl.cuda`, `bl.cpu`, or `bl.unified`). Defaults to `bl.cuda`.

    Returns:
        array: The created phasor tensor.
    """
    return _create_array("phasor_tensor", shape, dtype, device)

def tensor(shape, dtype=bl.f32, device=bl.cuda):
    """
    Creates a new tensor with the specified shape, data type, and device.

    Args:
        shape (tuple): The shape of the tensor.
        dtype (bl.constant, optional): The data type of the tensor (`bl.f32`, `bl.i8`, etc). Defaults to `bl.f32`.
        device (bl.constant, optional): The device to create the tensor on (`bl.cuda`, `bl.cpu`, or `bl.unified`). Defaults to `bl.cuda`.

    Returns:
        Tensor: A new tensor with the specified shape, data type, and device.
    """
    return _create_array("tensor", shape, dtype, device)