import importlib

import blade._internal.base as bl


def module(name, config, input, out=bl.cf32, telescope=bl.generic):
    _name = name.value
    _tmp = _name.split('-')
    _module_ext = _tmp[0]
    _module_taint = _tmp[1] if len(_tmp) > 1 else None
    _module_out_dtype = out.value
    _module_telescope = telescope.value if telescope != bl.generic else None
    _pipeline = bl._FetchPipeline()

    try:
        _module = importlib.import_module(f'blade._{_module_ext}_impl')
    except ModuleNotFoundError:
        raise ModuleNotFoundError(f"Can't find specified module extensions ({_module_ext}).")

    if _module_telescope:
        _telescope_list = [attr for attr in dir(_module) if attr in bl._telescope_lst]

        if _module_telescope not in _telescope_list:
            if len(_telescope_list) == 0:
                raise AttributeError(f"The module '{_name}' only supports generic telescopes.")
            else:
                raise AttributeError(f"The module '{_name}' only supports these telescopes: {', '.join(_telescope_list)}.")
    
    _caller = _module

    if _module_telescope:
        _caller = getattr(_caller, _module_telescope)

    if _module_taint:
        _caller = getattr(_caller, _module_taint)

    if _module_out_dtype:
        _caller = getattr(_caller, _module_out_dtype)

    _config = getattr(_caller, 'config')
    _input = getattr(_caller, 'input')

    if isinstance(config, tuple):
        _config = _config(*config)
    elif isinstance(config, dict):
        _config = _config(**config)
    elif isinstance(config, int):
        _config = _config(config)
    else:
        raise ValueError('Config should be a Tuple, Dict, or Int.')

    if isinstance(input, tuple):
        _input = _input(*input)
    elif isinstance(input, dict):
        _input = _input(**input)
    elif ('blade._mem_impl.cuda' in input.__class__.__module__) or \
         ('blade._mem_impl.cpu' in input.__class__.__module__):
        _input = _input(input)
    else:
        raise ValueError('Input should be a Tuple, Dict, or Vector.')
    
    # TODO: Handle bundles.
    # TODO: Send stream while calling.
    _module_inst = _caller(_config, _input)
    
    if _pipeline:
        _pipeline.add_module(_module_inst)

    return _module_inst