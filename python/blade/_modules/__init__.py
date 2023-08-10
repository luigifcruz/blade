import importlib as _importlib

import blade._internal.base as _bl

_extension_list = [
    'beamformer',
    'bfr5',
    'cast',
    'duplicate',
    'channelizer',
    'detector',
    'gather',
    'guppi',
    'permutation',
    'phasor',
    'dasdsa',
    'polarizer',
]

for _extension_name in _extension_list:
    try:
        _extension_import = _importlib.import_module(f'blade._{_extension_name}_impl')
        _extension_module_dict = {}
        for attr_name in dir(_extension_import):
            if not attr_name.startswith("__"):
                _extension_module_dict[attr_name] = getattr(_extension_import, attr_name)
        globals()[_extension_name] = _extension_module_dict
    except:
        pass

def module(type, config, input, out_type=_bl.cf32, telescope=_bl.generic):
    pass