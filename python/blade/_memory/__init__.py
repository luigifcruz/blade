import importlib

import blade._internal.base as bl


def _create_array(type, shape, dtype, device):
    _pipeline = bl._FetchPipeline()
    _type = type + '_duet' if _pipeline else type
    _shape = shape
    _dtype = dtype.value
    _device = device.value if device != bl.unified else bl.cuda.value
    _unified = False if device != bl.unified else True

    try:
        _mem = importlib.import_module(f'blade._mem_impl')
    except ModuleNotFoundError:
        raise ModuleNotFoundError(f"Can't find specified memory extension.")

    _caller = _mem

    _caller = getattr(_caller, _device)
    _caller = getattr(_caller, _dtype)
    _caller = getattr(_caller, _type)

    return _caller(_shape, unified=_unified)


def array_tensor(shape, dtype=bl.f32, device=bl.cuda):
    return _create_array("array_tensor", shape, dtype, device)

def phasor_tensor(shape, dtype=bl.f32, device=bl.cuda):
    return _create_array("phasor_tensor", shape, dtype, device)

def tensor(shape, dtype=bl.f32, device=bl.cuda):
    return _create_array("tensor", shape, dtype, device)
