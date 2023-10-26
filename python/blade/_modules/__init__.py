import importlib

import blade._internal as bl

def module(name, config, input, it=bl.cf32, ot=bl.cf32, telescope=bl.generic):
    """
    Dynamically loads a module extension implementation and returns an instance of it.

    Args:
        name (bl.constant): The name of the module to load.
        config (Union[Tuple, Dict, int]): The configuration for the module.
        input (Union[Tuple, Dict, Vector]): The input data for the module.
        it (bl.constant, optional): The input data type (`bl.cf32, `bl.f32`, etc). Defaults to `bl.cf32`.
        ot (bl.constant, optional): The output data type (`bl.cf32, `bl.f32`, etc). Defaults to `bl.cf32`.
        telescope (bl.constant, optional): The telescope type (`bl.generic, `bl.ata`, `bl.vla`, etc). Defaults to `bl.generic`.

    Returns:
        An instance of the loaded module extension implementation.

    Raises:
        ValueError: If the module name, output data type, or input data type is not a constant type.
        ModuleNotFoundError: If the specified module extension cannot be found.
        AttributeError: If the module does not support the specified telescope, taint, or output data type.
        ValueError: If the configuration or input data is not a Tuple, Dict, or Int.
        ValueError: If the input data is not a Tuple, Dict, or Vector.
        RuntimeError: If a new module is attempted to be connected after the pipeline is committed.
    """

    # Validate input parameters.
    if isinstance(name, bl._Constant):
        _name = name.value
    else:
        raise ValueError("Module name has to be a constant type (`bl.beamformer`, `bl.phasor`, etc).")
    
    if isinstance(it, bl._Constant):
        _in_type = it.value
    else:
        raise ValueError("Module input data type has to be a constant type (`bl.cf32`, `bl.f32`, etc).")

    if isinstance(ot, bl._Constant):
        _out_type = ot.value
    else:
        raise ValueError("Module output data type has to be a constant type (`bl.cf32`, `bl.f32`, etc).")

    if isinstance(telescope, bl._Constant):
        _telescope = telescope.value
    else:
        raise ValueError("Module telescope has to be a constant type (`bl.generic`, `bl.ata`, `bl.vla`, etc).")

    _tmp = _name.split("_")
    _ext_name = _tmp[0]
    _ext_taint = _tmp[1] if len(_tmp) > 1 else None
    _pipeline = bl._Fetch()

    # Import module extension implementation.
    try:
        _ext = importlib.import_module(f"blade._{_ext_name}_impl")
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
    
    # Check if module supports input data type.
    if any(s.startswith("in_") for s in dir(_caller)):
        _in_list = [s.replace("in_", "") for s in dir(_caller) if not s.startswith("_")]
        if _in_type not in _in_list:
            raise AttributeError(f"The module '{_name}' only supports these input types: {', '.join(_in_list)}.")
        _caller = getattr(_caller, f"in_{_in_type}")

    # Check if module supports output data type.
    if any(s.startswith("out_") for s in dir(_caller)):
        _out_list = [s.replace("out_", "") for s in dir(_caller) if not s.startswith("_")]
        if _out_type not in _out_list:
            raise AttributeError(f"The module '{_name}' with input '{_in_type}' only supports these output types: {', '.join(_out_list)}.")
        _caller = getattr(_caller, f"out_{_out_type}")

    # Get module class.
    if "mod" in dir(_caller):
        _caller = getattr(_caller, "mod")
    else:
        raise RuntimeError(f"The module '{_name}' doesn't have a valid module handle.")

    # Automatically cast configuration to correct type.
    _config_struct = getattr(_caller, "config")

    if isinstance(config, tuple):
        _config_struct = _config_struct(*config)
    elif isinstance(config, dict):
        _config_struct = _config_struct(**config)
    elif isinstance(config, int):
        _config_struct = _config_struct(config)
    else:
        raise ValueError("Config should be a Tuple, Dict, or Int.")

    # Automatically cast input to correct type.
    _input_struct = getattr(_caller, "input")
    _sanitized_input = bl._sanitize_duet(input)

    if isinstance(input, tuple):
        _input_struct = _input_struct(*_sanitized_input)
    elif isinstance(input, dict):
        _input_struct = _input_struct(**_sanitized_input)
    elif ("blade._mem_impl.cuda" in input.__class__.__module__) or \
         ("blade._mem_impl.cpu" in input.__class__.__module__):
        _input_struct = _input_struct(_sanitized_input)
    else:
        raise ValueError("Input should be a Tuple, Dict, or Vector.")

    # Register module into current pipeline.

    # This is the same logic implemented in 'pipeline.h'.
    # It's duplicated because it would be sub-optimal to
    # redefine thousands of modules types permutations.
    if _pipeline:
        if _pipeline.commited():
            raise RuntimeError("Can't connect new module after Pipeline is commited.")

        _inst = _caller(_config_struct, _input_struct, _pipeline.stream())

        if isinstance(_inst, bl._hidden.bundle):
            for _module in _inst.modules():
                _pipeline.add_module(_module)
        else:
            _pipeline.add_module(_inst)
    else:
        # Instantiate module in the default stream if outside a pipeline.
        _inst = _caller(_config_struct, _input_struct, bl.stream())

    return _inst
