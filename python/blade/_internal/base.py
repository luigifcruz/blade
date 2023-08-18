import inspect as _inspect

from blade._const_impl import *
import blade._hidden_impl as _hidden

#
# Macros
#

def _FetchPipeline():
    if 'pipeline' in globals():
        return globals()['pipeline']

    _caller_frame = _inspect.currentframe()

    for _ in range(3):
        if _caller_frame is None:
            break
        _caller_frame = _caller_frame.f_back

    if _caller_frame and 'self' in _caller_frame.f_locals:
        _instance = _caller_frame.f_locals['self']
        if hasattr(_instance, 'pipeline'):
            return getattr(_instance, 'pipeline')

    return None

#
# Constants
#

class _Constant:
    def __init__(self, value):
        self._value = value

    def __repr__(self):
        return f'Constant(value={self._value})'

    @property
    def value(self):
        return self._value

# Is defining those dynamically a big mistake?
# Will the compiler/debugger pick those up?
# I don't know, but I not feeling like defining them manually.
# If this turns out to be a bad idea, we could do it.
_telescope_lst = ['ata', 'vla', 'meerkat', 'generic']
_device_lst    = ['cpu', 'cuda', 'unified']
_types_lst     = ['f16', 'f32', 'f64', 'i8', 'i16', 'i32', 'i64', 'u8', 'u16', 'u32', 'u64',
                  'cf16', 'cf32', 'cf64', 'ci8', 'ci16', 'ci32', 'ci64', 'cu8', 'cu16', 'cu32', 'cu64']
_modules_lst   = ['beamformer', 'bfr5_reader', 'cast', 'channelizer', 'duplicate', 'detector', 'gather',
                  'guppi_reader', 'guppi_writer', 'permutation', 'phasor', 'polarizer', 'modeH', 'modeB']

def _create_constants(names_list):
    for name in names_list:
        globals()[name] = _Constant(name)

_create_constants(_telescope_lst)
_create_constants(_device_lst)
_create_constants(_types_lst)
_create_constants(_modules_lst)
