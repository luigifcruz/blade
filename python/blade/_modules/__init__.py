import importlib

import blade._internal.base as bl


def module(name, config, input, out=bl.cf32, telescope=bl.generic):
    # Validate input parameters.
    if isinstance(name, bl._Constant):
        _name = name.value
    else:
        raise ValueError("Module name has to be a constant type (bl.beamformer, bl.phasor, etc).")

    if isinstance(out, bl._Constant):
        _out = out.value
    else:
        raise ValueError("Module output data type has to be a constant type (bl.cf32, bl.f32, etc).")

    if isinstance(telescope, bl._Constant):
        _telescope = telescope.value
    else:
        raise ValueError("Module telescope has to be a constant type (bl.generic, bl.ata, bl.vla, etc).")

    _input = input
    _config = config
    _tmp = _name.split('_')
    _ext_name = _tmp[0]
    _ext_taint = _tmp[1] if len(_tmp) > 1 else None
    _pipeline = bl._FetchPipeline()

    # Import module extension implementation.
    try:
        _ext = importlib.import_module(f'blade._{_ext_name}_impl')
    except ModuleNotFoundError:
        raise ModuleNotFoundError(f"Can't find specified module extension ({_ext_name}).")

    # Copy extension into caller.
    _caller = _ext

    # Check if module supports telescopes.
    if any(s.startswith("tel_") for s in dir(_caller)):
        _telescope_list = [attr.replace("tel_", "") for attr in dir(_caller) if attr.replace("tel_", "") in bl._telescope_lst]
        if _telescope not in _telescope_list:
            raise AttributeError(f"The module '{_name}' only supports these telescopes: {', '.join(_telescope_list)}.")
        _caller = getattr(_caller, f"tel_{_telescope}")
    elif _telescope != "generic":
        raise AttributeError(f"The module '{_name}' only supports generic telescopes (telescope=bl.generic).")

    # Check if module supports taints.
    if any(s.startswith("taint_") for s in dir(_caller)):
        _taint_list = [s.replace("taint_", "") for s in dir(_caller) if not s.startswith("_")]
        if _ext_taint not in _taint_list or not _ext_taint:
            raise AttributeError(f"The module '{_name}' only supports these taints: {', '.join(_taint_list)}.")
        _caller = getattr(_caller, f"taint_{_ext_taint}")
    elif _ext_taint:
        raise AttributeError(f"The module '{_name}' doesn't support taints.")

    # Check if module supports output data type.
    if any(s.startswith("type_") for s in dir(_caller)):
        _type_list = [s.replace("type_", "") for s in dir(_caller) if not s.startswith("_")]
        if _out not in _type_list:
            raise AttributeError(f"The module '{_name}' only supports these output types: {', '.join(_type_list)}.")
        _caller = getattr(_caller, f"type_{_out}")

    # Automatically cast configuration to correct type.
    _config = getattr(_caller, 'config')

    if isinstance(config, tuple):
        _config = _config(*config)
    elif isinstance(config, dict):
        _config = _config(**config)
    elif isinstance(config, int):
        _config = _config(config)
    else:
        raise ValueError('Config should be a Tuple, Dict, or Int.')

    # Automatically cast input to correct type.
    _input = getattr(_caller, 'input')

    if isinstance(input, tuple):
        _input = _input(*input)
    elif isinstance(input, dict):
        _input = _input(**input)
    elif ('blade._mem_impl.cuda' in input.__class__.__module__) or \
         ('blade._mem_impl.cpu' in input.__class__.__module__):
        _input = _input(input)
    else:
        raise ValueError('Input should be a Tuple, Dict, or Vector.')

    # Register module into current pipeline.

    # This is the same logic implemented in 'pipeline.h'.
    # It's duplicated because it would be a pain in the ass to
    # redefine all the modules types permutations.
    if _pipeline:
        if _pipeline.commited():
            raise RuntimeError("Can't connect new module after Pipeline is commited.")

        _inst = _caller(_config, _input, _pipeline.stream())

        if isinstance(_inst, bl._hidden.bundle):
            for _module in _inst.modules():
                _pipeline.add_module(_inst)
        else:
            _pipeline.add_module(_inst)
    else:
        # Instantiate module in the default stream if outside a pipeline.
        _inst = _caller(_config, _input, bl.stream())

    return _inst
